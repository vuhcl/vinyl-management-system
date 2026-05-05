"""
grader/src/models/baseline.py

Two-head logistic regression baseline for vinyl condition grading.
One head per target (sleeve, media) trained on TF-IDF features.

Pipeline:
  1. Load pre-built TF-IDF features from artifacts/features/
  2. Fit two LogisticRegression heads on train split
  3. Calibrate probabilities on val split using isotonic regression
  4. Evaluate on val and test splits
  5. Save fitted models and log metrics to MLflow

Rule engine post-processing is NOT applied here.
That responsibility belongs to pipeline.py.
This module is a pure model training and evaluation unit.

Output artifacts:
  grader/artifacts/baseline_sleeve.pkl
  grader/artifacts/baseline_media.pkl
  grader/artifacts/baseline_sleeve_calibrated.pkl
  grader/artifacts/baseline_media_calibrated.pkl
  grader/artifacts/confusion_matrix_sleeve.txt
  grader/artifacts/confusion_matrix_media.txt

Usage:
    python -m grader.src.models.baseline
    python -m grader.src.models.baseline --dry-run
"""

import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from grader.src.config_io import load_yaml_mapping
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    mlflow_start_run_ctx,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

TARGETS = ["sleeve", "media"]
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# _IsotonicCalibrator
# ---------------------------------------------------------------------------
class _IsotonicCalibrator:
    """
    Post-hoc probability calibrator using per-class isotonic regression.

    Wraps a fitted sklearn classifier and applies a separate isotonic
    regression to each class column of its predict_proba output.

    Critically, this calibrator always outputs ``n_classes`` columns where
    ``n_classes`` is the total number of canonical classes (passed in at
    construction time). If the base model was trained without some rare
    class (e.g. sleeve Excellent never appears in training data), those
    columns stay at zero so downstream blending logic sees a consistent
    shape.

    Drop-in replacement for CalibratedClassifierCV(cv="prefit") which was
    removed in sklearn 1.6+.
    """

    def __init__(self, base_clf, n_classes: int) -> None:
        self.base_clf = base_clf
        self.n_classes = n_classes
        self.calibrators_: list[Optional[IsotonicRegression]] = []
        self.classes_ = None

    def fit(self, X, y) -> "_IsotonicCalibrator":
        self.classes_ = self.base_clf.classes_   # model's actual classes (may be < n_classes)
        model_classes = list(self.classes_)
        raw_proba = self.base_clf.predict_proba(X)  # (n_samples, len(model_classes))

        # One-hot encode y against the model's actual class list
        y_onehot = np.zeros((len(y), len(model_classes)), dtype=float)
        cls_to_col = {c: i for i, c in enumerate(model_classes)}
        for row, label in enumerate(y):
            if label in cls_to_col:
                y_onehot[row, cls_to_col[label]] = 1.0

        # One calibrator per model class
        self.calibrators_ = []
        for j in range(len(model_classes)):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_proba[:, j], y_onehot[:, j])
            self.calibrators_.append(iso)
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw_proba = self.base_clf.predict_proba(X)
        model_classes = list(self.base_clf.classes_)

        # Build output with the full n_classes columns, mapping model classes
        # to their canonical integer indices (model class values are ints from
        # the LabelEncoder, so they directly index the output columns).
        n_samples = raw_proba.shape[0]
        out = np.zeros((n_samples, self.n_classes), dtype=float)
        for j, cls_idx in enumerate(model_classes):
            out[:, int(cls_idx)] = self.calibrators_[j].predict(raw_proba[:, j])

        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return out / row_sum


# ---------------------------------------------------------------------------
# BaselineModel
# ---------------------------------------------------------------------------
class BaselineModel:
    """
    Two-head logistic regression classifier for vinyl condition grading.

    Each head is an independent LogisticRegression fitted on TF-IDF
    features for a single target (sleeve or media). Probability outputs
    are calibrated using isotonic regression on the val split.

    Config keys read from grader.yaml:
        models.baseline.logistic_regression.*  — LR hyperparameters
        evaluation.calibration.method          — calibration method
        paths.artifacts                         — artifact directory
        mlflow (URI from MLFLOW_TRACKING_URI / tracking_uri_fallback)
        mlflow.experiment_name
    """

    def __init__(
        self,
        config_path: str,
        config: Optional[dict] = None,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = load_yaml_mapping(config_path)
        self._config_path_str = config_path

        lr_cfg = self.config["models"]["baseline"]["logistic_regression"]
        self.C: float = lr_cfg["C"]
        self.max_iter: int = lr_cfg["max_iter"]
        self.class_weight: str = lr_cfg["class_weight"]
        self.solver: str = lr_cfg["solver"]
        self.random_state: int = lr_cfg.get("random_state", 42)
        self.tol: float = float(lr_cfg.get("tol", 1e-4))
        tuning_cfg = self.config["models"]["baseline"].get("tuning", {})
        self.tuning_enabled: bool = bool(tuning_cfg.get("enabled", False))
        self.tuning_c_values: list[float] = [
            float(v) for v in tuning_cfg.get("c_values", [self.C])
        ]
        boundary_cfg = self.config["models"]["baseline"].get(
            "boundary_objective", {}
        )
        self.boundary_enabled: bool = bool(boundary_cfg.get("enabled", False))
        self.boundary_alpha: float = float(boundary_cfg.get("alpha", 0.2))
        band_cfg = self.config["models"]["baseline"].get("confidence_band", {})
        self.confidence_band_enabled: bool = bool(band_cfg.get("enabled", True))
        self.confidence_band_max_gap: float = float(
            band_cfg.get("max_gap", 0.12)
        )
        gating_cfg = self.config["models"]["baseline"].get(
            "media_evidence_gating", {}
        )
        self.media_gating_enabled: bool = bool(gating_cfg.get("enabled", False))
        self.media_gating_alpha_none: float = float(
            gating_cfg.get("alpha_none", 0.35)
        )
        self.media_gating_alpha_weak: float = float(
            gating_cfg.get("alpha_weak", 0.20)
        )
        aux_cfg = self.config["models"]["baseline"].get(
            "media_evidence_aux", {}
        )
        self.media_evidence_aux_enabled: bool = bool(
            aux_cfg.get("enabled", False)
        )

        cal_cfg = self.config["evaluation"]["calibration"]
        self.calibration_method: str = cal_cfg.get("method", "isotonic")

        # Paths
        self.artifacts_dir = Path(self.config["paths"]["artifacts"])
        self.features_dir = self.artifacts_dir / "features"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.model_paths = {
            target: self.artifacts_dir / f"baseline_{target}.pkl"
            for target in TARGETS
        }
        self.calibrated_paths = {
            target: self.artifacts_dir / f"baseline_{target}_calibrated.pkl"
            for target in TARGETS
        }
        self.confusion_paths = {
            target: self.artifacts_dir / f"confusion_matrix_{target}.txt"
            for target in TARGETS
        }

        # Fitted objects — populated during run()
        self.models: dict[str, LogisticRegression] = {}
        self.calibrated: dict[str, "_IsotonicCalibrator"] = {}
        self.boundary_models: dict[str, LogisticRegression] = {}
        self.boundary_band_labels: dict[str, np.ndarray] = {}
        self.media_evidence_aux_model: Optional[LogisticRegression] = None
        self.encoders: dict = {}
        self.selected_c: dict[str, float] = {}
        self.grade_ordinal_map = self._load_grade_ordinal_map()

        if mlflow_enabled(self.config):
            configure_mlflow_from_config(self.config)

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    def _load_grade_ordinal_map(self) -> dict[str, int]:
        rules_cfg = self.config.get("rules", {})
        guidelines_path = rules_cfg.get("guidelines_path")
        if not guidelines_path:
            return {}
        p = Path(guidelines_path)
        if not p.exists():
            return {}
        g = load_yaml_mapping(p)
        return {
            str(k): int(v)
            for k, v in g.get("grade_ordinal_map", {}).items()
            if isinstance(v, int)
        }

    # -----------------------------------------------------------------------
    # Feature loading
    # -----------------------------------------------------------------------
    def load_all_features(
        self,
    ) -> dict[str, dict[str, dict]]:
        """
        Load TF-IDF feature matrices and label arrays for all
        splits and targets from artifacts/features/.

        Returns nested dict: features[split][target] = {"X": ..., "y": ...}
        """
        features: dict = {split: {} for split in SPLITS}

        for split in SPLITS:
            for target in TARGETS:
                X, y = TFIDFFeatureBuilder.load_features(
                    str(self.features_dir), split, target
                )
                features[split][target] = {"X": X, "y": y}
                logger.info(
                    "Loaded features — split=%s target=%s shape=%s",
                    split,
                    target,
                    X.shape,
                )

        thin_x = self.features_dir / "test_thin_sleeve_X.npz"
        if thin_x.exists():
            features["test_thin"] = {}
            for target in TARGETS:
                X, y = TFIDFFeatureBuilder.load_features(
                    str(self.features_dir), "test_thin", target
                )
                features["test_thin"][target] = {"X": X, "y": y}
                logger.info(
                    "Loaded features — split=test_thin target=%s shape=%s",
                    target,
                    X.shape,
                )

        return features

    def load_encoders(self) -> dict:
        """Load fitted label encoders from artifacts/."""
        encoders = {}
        for target in TARGETS:
            path = self.artifacts_dir / f"label_encoder_{target}.pkl"
            encoders[target] = TFIDFFeatureBuilder.load_encoder(str(path))
            logger.info(
                "Loaded label encoder — target=%s classes=%s",
                target,
                list(encoders[target].classes_),
            )
        return encoders

    # -----------------------------------------------------------------------
    # Model construction and training
    # -----------------------------------------------------------------------
    def build_model(
        self,
        target: str,
        *,
        C: Optional[float] = None,
    ) -> LogisticRegression:
        """
        Construct a LogisticRegression head for a single target.
        Hyperparameters come from grader.yaml — nothing hardcoded.
        """
        return LogisticRegression(
            C=self.C if C is None else C,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )

    @staticmethod
    def _macro_f1(y_true, y_pred) -> float:
        return float(f1_score(y_true, y_pred, average="macro"))

    def _tune_c_for_target(self, target: str, features: dict) -> float:
        """
        Grid-search C on train->val for a single target.
        Uses macro-F1 on val as selection metric.
        """
        X_train = features["train"][target]["X"]
        y_train = features["train"][target]["y"]
        X_val = features["val"][target]["X"]
        y_val = features["val"][target]["y"]

        best_c = self.C
        best_val_f1 = -1.0

        for c in self.tuning_c_values:
            model = self.build_model(target, C=c)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_f1 = self._macro_f1(y_val, val_pred)
            logger.info(
                "Tune baseline C — target=%s C=%.4f val macro-F1=%.4f",
                target,
                c,
                val_f1,
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_c = c

        logger.info(
            "Selected baseline C — target=%s C=%.4f (best val macro-F1=%.4f)",
            target,
            best_c,
            best_val_f1,
        )
        return best_c

    def train(
        self,
        features: dict,
    ) -> dict[str, LogisticRegression]:
        """
        Fit one LogisticRegression head per target on train features.

        Args:
            features: nested dict from load_all_features()

        Returns:
            Dict mapping target → fitted LogisticRegression.
        """
        models = {}
        for target in TARGETS:
            logger.info("Training baseline head — target=%s ...", target)
            X_train = features["train"][target]["X"]
            y_train = features["train"][target]["y"]

            selected_c = self.C
            if self.tuning_enabled:
                selected_c = self._tune_c_for_target(target, features)
            self.selected_c[target] = selected_c

            model = self.build_model(target, C=selected_c)
            model.fit(X_train, y_train)
            models[target] = model

            train_preds = model.predict(X_train)
            train_f1 = f1_score(y_train, train_preds, average="macro")
            logger.info("Train macro-F1 — target=%s: %.4f", target, train_f1)

        return models

    @staticmethod
    def _grade_to_band(grade: str) -> str:
        if grade in {"Poor", "Good"}:
            return "low"
        if grade in {"Very Good", "Very Good Plus"}:
            return "mid"
        if grade in {"Excellent", "Near Mint", "Mint"}:
            return "high"
        return "other"

    def train_boundary_models(self, features: dict) -> None:
        """Train coarse-bin auxiliary heads for boundary-aware blending."""
        if not self.boundary_enabled:
            return
        self.boundary_models = {}
        self.boundary_band_labels = {}
        for target in TARGETS:
            X_train = features["train"][target]["X"]
            y_train = features["train"][target]["y"]
            encoder = self.encoders[target]
            grades = encoder.inverse_transform(y_train)
            band_labels = np.asarray([self._grade_to_band(g) for g in grades])
            bmodel = LogisticRegression(
                C=self.selected_c.get(target, self.C),
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=self.class_weight,
                solver=self.solver,
                random_state=self.random_state,
            )
            bmodel.fit(X_train, band_labels)
            self.boundary_models[target] = bmodel
            self.boundary_band_labels[target] = bmodel.classes_
            logger.info(
                "Boundary auxiliary head trained — target=%s bands=%s",
                target,
                list(bmodel.classes_),
            )

    def train_media_evidence_aux(
        self,
        features: dict,
        split_records: dict[str, list[dict]],
    ) -> None:
        """Train auxiliary classifier for media evidence strength."""
        if not self.media_evidence_aux_enabled:
            self.media_evidence_aux_model = None
            return
        train_records = split_records.get("train", [])
        X_train = features["train"]["media"]["X"]
        y = np.asarray(
            [
                str(r.get("media_evidence_strength", "none")).lower()
                for r in train_records
            ]
        )
        if len(y) != X_train.shape[0]:
            logger.warning(
                "Skipping media evidence aux head: record/feature mismatch."
            )
            self.media_evidence_aux_model = None
            return
        model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight="balanced",
            solver=self.solver,
            random_state=self.random_state,
        )
        model.fit(X_train, y)
        self.media_evidence_aux_model = model
        logger.info(
            "Media evidence aux head trained — classes=%s",
            list(model.classes_),
        )

    def _blend_with_boundary(
        self,
        target: str,
        X,
        base_proba: np.ndarray,
    ) -> np.ndarray:
        if not self.boundary_enabled or target not in self.boundary_models:
            return base_proba
        encoder = self.encoders[target]
        n_classes = len(encoder.classes_)
        band_model = self.boundary_models[target]
        band_proba = band_model.predict_proba(X)
        band_classes = list(band_model.classes_)
        mapped = np.zeros((base_proba.shape[0], n_classes), dtype=float)
        for j, grade in enumerate(encoder.classes_):
            band = self._grade_to_band(str(grade))
            if band in band_classes:
                mapped[:, j] = band_proba[:, band_classes.index(band)]
        blended = (1.0 - self.boundary_alpha) * base_proba + (
            self.boundary_alpha * mapped
        )
        # base_proba already has n_classes columns (from _IsotonicCalibrator)
        assert base_proba.shape[1] == n_classes, (
            f"base_proba has {base_proba.shape[1]} cols but encoder has {n_classes}"
        )
        row_sum = blended.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return blended / row_sum

    def _predict_with_proba(self, target: str, X) -> tuple[np.ndarray, np.ndarray]:
        proba = self.calibrated[target].predict_proba(X)
        proba = self._blend_with_boundary(target, X, proba)
        pred = proba.argmax(axis=1)
        return pred, proba

    def _apply_media_evidence_gating(
        self,
        proba: np.ndarray,
        media_strengths: Optional[list[str]],
    ) -> np.ndarray:
        if not self.media_gating_enabled or media_strengths is None:
            return proba
        classes = list(self.encoders["media"].classes_)
        conservative = {"Near Mint", "Very Good Plus", "Very Good"}
        prior = np.zeros(len(classes), dtype=np.float64)
        for i, cls in enumerate(classes):
            if cls in conservative:
                prior[i] = 1.0
        if prior.sum() == 0:
            return proba
        prior = prior / prior.sum()
        out = proba.copy()
        n = min(len(media_strengths), out.shape[0])
        for i in range(n):
            s = str(media_strengths[i]).lower()
            if s == "none":
                a = self.media_gating_alpha_none
            elif s == "weak":
                a = self.media_gating_alpha_weak
            else:
                a = 0.0
            if a > 0:
                out[i, :] = (1.0 - a) * out[i, :] + a * prior
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return out / row_sum

    def _confidence_band(self, scores: dict[str, float]) -> Optional[str]:
        if not self.confidence_band_enabled or len(scores) < 2:
            return None
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (g1, p1), (g2, p2) = ranked[0], ranked[1]
        if (p1 - p2) > self.confidence_band_max_gap:
            return None
        if g1 not in self.grade_ordinal_map or g2 not in self.grade_ordinal_map:
            return None
        if abs(self.grade_ordinal_map[g1] - self.grade_ordinal_map[g2]) != 1:
            return None
        ordered = sorted([g1, g2], key=lambda g: self.grade_ordinal_map[g])
        return f"{ordered[0]}/{ordered[1]}"

    # -----------------------------------------------------------------------
    # Calibration — fitted on val split
    # -----------------------------------------------------------------------
    def calibrate(
        self,
        models: dict[str, LogisticRegression],
        features: dict,
    ) -> dict[str, "_IsotonicCalibrator"]:
        """
        Calibrate each fitted model on the val split using per-class isotonic
        regression on the base model's raw predict_proba output.

        This approach preserves the full class set from the base estimator
        (including rare grades like Excellent that may not appear in val),
        avoiding the class-mismatch crash that occurs with CalibratedClassifierCV
        when val lacks some training classes.

        Calibration is fitted on val — NOT train — to prevent overfitting.

        Args:
            models:   fitted LogisticRegression heads
            features: nested dict from load_all_features()

        Returns:
            Dict mapping target → _IsotonicCalibrator wrapper.
        """
        calibrated = {}
        for target in TARGETS:
            logger.info(
                "Calibrating model — target=%s method=%s ...",
                target,
                self.calibration_method,
            )
            X_val = features["val"][target]["X"]
            y_val = features["val"][target]["y"]

            n_classes = len(self.encoders[target].classes_)
            calibrator = _IsotonicCalibrator(models[target], n_classes=n_classes)
            calibrator.fit(X_val, y_val)
            calibrated[target] = calibrator

            logger.info("Calibration complete — target=%s", target)

        return calibrated

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------
    def predict(
        self,
        X_sleeve,
        X_media,
        item_ids: Optional[list[str]] = None,
        records: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Run inference on feature matrices and return structured
        predictions conforming to the agreed output schema.

        Rule engine is NOT applied here — that is pipeline.py's job.
        rule_override_applied is always False at this stage.

        Args:
            X_sleeve:  sparse feature matrix for sleeve target
            X_media:   sparse feature matrix for media target
            item_ids:  optional list of item IDs for output
            records:   optional list of source records for metadata

        Returns:
            List of prediction dicts with confidence scores.
        """
        if not self.calibrated:
            raise RuntimeError(
                "Model not calibrated. Run train() and calibrate() first."
            )

        n_samples = X_sleeve.shape[0]
        if item_ids is None:
            item_ids = [str(i) for i in range(n_samples)]

        sleeve_encoder = self.encoders["sleeve"]
        media_encoder = self.encoders["media"]

        # Predicted class indices and probabilities
        sleeve_pred_idx, sleeve_proba = self._predict_with_proba(
            "sleeve", X_sleeve
        )
        media_pred_idx, media_proba = self._predict_with_proba("media", X_media)
        media_strengths = None
        if records:
            media_strengths = [
                str(r.get("media_evidence_strength", "none")) for r in records
            ]
        media_proba = self._apply_media_evidence_gating(
            media_proba,
            media_strengths,
        )
        media_pred_idx = media_proba.argmax(axis=1)

        # Decode integer indices back to grade strings
        sleeve_pred = sleeve_encoder.inverse_transform(sleeve_pred_idx)
        media_pred = media_encoder.inverse_transform(media_pred_idx)

        sleeve_classes = sleeve_encoder.classes_
        media_classes = media_encoder.classes_

        predictions = []
        for i in range(n_samples):
            # Extract metadata from source record if available
            record = records[i] if records else {}
            media_verifiable = record.get("media_verifiable", True)
            contradiction = record.get("contradiction_detected", False)
            source = record.get("source", "unknown")
            media_strength = record.get("media_evidence_strength", "none")

            # Build confidence score dicts
            sleeve_scores = {
                cls: round(float(sleeve_proba[i][j]), 4)
                for j, cls in enumerate(sleeve_classes)
            }
            media_scores = {
                cls: round(float(media_proba[i][j]), 4)
                for j, cls in enumerate(media_classes)
            }
            sleeve_band = self._confidence_band(sleeve_scores)
            media_band = self._confidence_band(media_scores)
            excellent_proxy_media = 0.0
            nm_key = "Near Mint"
            vgp_key = "Very Good Plus"
            if nm_key in media_scores and vgp_key in media_scores:
                excellent_proxy_media = round(
                    2.0 * min(media_scores[nm_key], media_scores[vgp_key]),
                    4,
                )
            evidence_scores: dict[str, float] = {}
            if self.media_evidence_aux_model is not None:
                eproba = self.media_evidence_aux_model.predict_proba(
                    X_media[i : i + 1]
                )[0]
                for j, cls in enumerate(self.media_evidence_aux_model.classes_):
                    evidence_scores[str(cls)] = round(float(eproba[j]), 4)

            predictions.append(
                {
                    "item_id": item_ids[i],
                    "predicted_sleeve_condition": str(sleeve_pred[i]),
                    "predicted_media_condition": str(media_pred[i]),
                    "confidence_scores": {
                        "sleeve": sleeve_scores,
                        "media": media_scores,
                    },
                    "metadata": {
                        "source": source,
                        "media_verifiable": media_verifiable,
                        "media_evidence_strength": media_strength,
                        "confidence_band_sleeve": sleeve_band,
                        "confidence_band_media": media_band,
                        "excellent_proxy_media": excellent_proxy_media,
                        "media_evidence_scores": evidence_scores,
                        "ambiguous_prediction": bool(
                            sleeve_band is not None or media_band is not None
                        ),
                        "rule_override_applied": False,
                        "rule_override_target": None,
                        "contradiction_detected": contradiction,
                    },
                }
            )

        return predictions

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    def evaluate(
        self,
        features: dict,
        split: str,
        records: Optional[list[dict]] = None,
    ) -> dict[str, dict]:
        """
        Compute evaluation metrics for both targets on a given split.

        Metrics computed:
          - Macro-F1 (primary)
          - Accuracy
          - Per-class F1, precision, recall
          - Expected Calibration Error (ECE)

        Args:
            features: nested dict from load_all_features()
            split:    "train", "val", "test", or "test_thin" (if features exist)

        Returns:
            Dict mapping target → metrics dict.
        """
        results = {}

        for target in TARGETS:
            X = features[split][target]["X"]
            y_true = features[split][target]["y"]
            encoder = self.encoders[target]

            y_pred, y_proba = self._predict_with_proba(target, X)
            if target == "media":
                strengths = None
                if records is not None:
                    strengths = [
                        str(r.get("media_evidence_strength", "none"))
                        for r in records
                    ]
                y_proba = self._apply_media_evidence_gating(y_proba, strengths)
                y_pred = y_proba.argmax(axis=1)

            macro_f1 = f1_score(y_true, y_pred, average="macro")
            accuracy = accuracy_score(y_true, y_pred)
            ece = self._compute_ece(y_true, y_proba)

            logger.info(
                "Evaluation — split=%s target=%s | "
                "macro-F1: %.4f | accuracy: %.4f | ECE: %.4f",
                split,
                target,
                macro_f1,
                accuracy,
                ece,
            )

            # Per-class report
            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(len(encoder.classes_))),
                target_names=list(encoder.classes_),
                output_dict=True,
                zero_division=0,
            )

            results[target] = {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "ece": ece,
                "report": report,
            }

        return results

    @staticmethod
    def _compute_ece(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        Measures how well predicted probabilities match actual frequencies.
        Lower is better. 0.0 is perfectly calibrated.

        Uses confidence of predicted class (max probability) for binning.
        """
        confidences = y_proba.max(axis=1)
        correct = (y_proba.argmax(axis=1) == y_true).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if not np.any(mask):
                continue
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    # -----------------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------------
    def format_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target: str,
    ) -> str:
        """Format a human-readable confusion matrix string."""
        encoder = self.encoders[target]
        classes = encoder.classes_
        # Pass all canonical label indices so cm is always n_classes × n_classes
        # even when rare classes (e.g. Excellent) are absent from the test split.
        all_labels = list(range(len(classes)))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        col_width = max(len(c) for c in classes) + 2
        header = " " * col_width + "".join(
            f"{c:>{col_width}}" for c in classes
        )
        lines = [
            f"CONFUSION MATRIX — {target.upper()}",
            "-" * len(header),
            header,
            "-" * len(header),
        ]
        for i, row_label in enumerate(classes):
            row = f"{row_label:<{col_width}}" + "".join(
                f"{cm[i][j]:>{col_width}}" for j in range(len(classes))
            )
            lines.append(row)
        lines.append("")
        return "\n".join(lines)

    def save_confusion_matrices(self, features: dict) -> None:
        """Save confusion matrices for both targets on test split."""
        for target in TARGETS:
            X = features["test"][target]["X"]
            y_true = features["test"][target]["y"]
            y_pred, _ = self._predict_with_proba(target, X)

            cm_text = self.format_confusion_matrix(y_true, y_pred, target)
            with open(self.confusion_paths[target], "w") as f:
                f.write(cm_text)
            logger.info("Saved confusion matrix — target=%s", target)

    # -----------------------------------------------------------------------
    # Artifact persistence
    # -----------------------------------------------------------------------
    def save_models(self) -> None:
        """Save both raw and calibrated models to artifacts/."""
        for target in TARGETS:
            with open(self.model_paths[target], "wb") as f:
                pickle.dump(self.models[target], f)
            with open(self.calibrated_paths[target], "wb") as f:
                pickle.dump(self.calibrated[target], f)
            logger.info("Saved models — target=%s", target)

    @staticmethod
    def load_model(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_trained_from_artifacts(cls, config_path: str) -> Tuple["BaselineModel", dict]:
        """
        Load pickled baseline heads + encoders from ``paths.artifacts`` and
        evaluate train/val/test using on-disk TF-IDF features.

        Used when skipping training (e.g. Colab after local baseline train).
        Expects the same artifacts as ``run()`` would write.
        """
        inst = cls(config_path=config_path)
        inst.encoders = inst.load_encoders()
        for target in TARGETS:
            raw_path = inst.model_paths[target]
            cal_path = inst.calibrated_paths[target]
            if not raw_path.is_file() or not cal_path.is_file():
                raise FileNotFoundError(
                    f"Missing baseline artifact(s) for {target}: "
                    f"{raw_path} and/or {cal_path} — train baseline locally first "
                    "or copy artifacts into paths.artifacts."
                )
            inst.models[target] = cls.load_model(str(raw_path))
            inst.calibrated[target] = cls.load_model(str(cal_path))

        features = inst.load_all_features()
        split_records: dict[str, list[dict]] = {}
        splits_dir = Path(inst.config["paths"]["splits"])
        for split in SPLITS:
            path = splits_dir / f"{split}.jsonl"
            records: list[dict] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            split_records[split] = records
        if "test_thin" in features:
            tt_path = splits_dir / "test_thin.jsonl"
            thin_recs: list[dict] = []
            if tt_path.exists():
                with open(tt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            thin_recs.append(json.loads(line))
            split_records["test_thin"] = thin_recs

        eval_results: dict[str, dict[str, dict]] = {}
        for split in SPLITS:
            eval_results[split] = inst.evaluate(
                features,
                split,
                records=split_records.get(split),
            )
        if "test_thin" in features:
            eval_results["test_thin"] = inst.evaluate(
                features,
                "test_thin",
                records=split_records.get("test_thin"),
            )

        bundle = {
            "models": inst.models,
            "calibrated": inst.calibrated,
            "eval": eval_results,
        }
        thin_note = (
            ", ".join((*SPLITS, "test_thin"))
            if "test_thin" in eval_results
            else ", ".join(SPLITS)
        )
        logger.info(
            "Loaded baseline from artifacts — evaluated %s (no training).",
            thin_note,
        )
        return inst, bundle

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(
        self,
        eval_results: dict[str, dict[str, dict]],
    ) -> None:
        # Parameters
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "lr_C": self.C,
                "lr_max_iter": self.max_iter,
                "lr_tol": self.tol,
                "lr_class_weight": self.class_weight,
                "lr_solver": self.solver,
                "calibration_method": self.calibration_method,
                "tuning_enabled": self.tuning_enabled,
                "boundary_enabled": self.boundary_enabled,
                "boundary_alpha": self.boundary_alpha,
                "confidence_band_enabled": self.confidence_band_enabled,
                "media_gating_enabled": self.media_gating_enabled,
                "media_gating_alpha_none": self.media_gating_alpha_none,
                "media_gating_alpha_weak": self.media_gating_alpha_weak,
                "media_evidence_aux_enabled": self.media_evidence_aux_enabled,
            }
        )
        for target in TARGETS:
            mlflow.log_param(
                f"selected_C_{target}",
                self.selected_c.get(target, self.C),
            )

        # Metrics per split per target
        for split, target_results in eval_results.items():
            for target, metrics in target_results.items():
                prefix = f"{split}_{target}"
                mlflow.log_metrics(
                    {
                        f"{prefix}_macro_f1": metrics["macro_f1"],
                        f"{prefix}_accuracy": metrics["accuracy"],
                        f"{prefix}_ece": metrics["ece"],
                    }
                )
                # Per-class F1 on primary test splits
                if split in ("test", "test_thin"):
                    for class_name, class_metrics in metrics["report"].items():
                        if isinstance(class_metrics, dict):
                            clean = class_name.lower().replace(" ", "_")
                            mlflow.log_metric(
                                f"{split}_{target}_{clean}_f1",
                                class_metrics.get("f1-score", 0.0),
                            )

        if mlflow_log_artifacts_enabled(self.config):
            for target in TARGETS:
                mlflow.log_artifact(str(self.calibrated_paths[target]))
                mlflow.log_artifact(str(self.confusion_paths[target]))

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> dict:
        """
        Full baseline training pipeline:
          1. Load TF-IDF features and label encoders
          2. Train two-head logistic regression on train split
          3. Calibrate on val split
          4. Evaluate on train, val, and test splits
          5. Save confusion matrices
          6. Save model artifacts
          7. Log all metrics and artifacts to MLflow

        Args:
            dry_run: train and evaluate but do not save artifacts
                     or log to MLflow.

        Returns:
            Dict with models, calibrated models, and eval results.
        """
        with mlflow_start_run_ctx(self.config, "baseline_tfidf_logreg"):

            # Load features and encoders
            features = self.load_all_features()
            self.encoders = self.load_encoders()

            # Train
            self.models = self.train(features)
            self.train_boundary_models(features)

            split_records: dict[str, list[dict]] = {}
            splits_dir = Path(self.config["paths"]["splits"])
            for split in SPLITS:
                path = splits_dir / f"{split}.jsonl"
                records: list[dict] = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                split_records[split] = records
            if "test_thin" in features:
                tt_path = splits_dir / "test_thin.jsonl"
                thin_recs: list[dict] = []
                if tt_path.exists():
                    with open(tt_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                thin_recs.append(json.loads(line))
                split_records["test_thin"] = thin_recs
            self.train_media_evidence_aux(features, split_records)

            # Calibrate on val
            self.calibrated = self.calibrate(self.models, features)

            # Evaluate on all splits
            eval_results: dict[str, dict] = {}
            for split in SPLITS:
                logger.info("--- Evaluating on %s split ---", split.upper())
                eval_results[split] = self.evaluate(
                    features,
                    split,
                    records=split_records.get(split),
                )
            if "test_thin" in features:
                logger.info("--- Evaluating on TEST_THIN split ---")
                eval_results["test_thin"] = self.evaluate(
                    features,
                    "test_thin",
                    records=split_records.get("test_thin"),
                )

            # Summary log
            for target in TARGETS:
                msg = (
                    "RESULTS SUMMARY — target=%s | "
                    "train macro-F1: %.4f | "
                    "val macro-F1: %.4f | "
                    "test macro-F1: %.4f"
                ) % (
                    target,
                    eval_results["train"][target]["macro_f1"],
                    eval_results["val"][target]["macro_f1"],
                    eval_results["test"][target]["macro_f1"],
                )
                if "test_thin" in eval_results:
                    msg += " | test_thin macro-F1: %.4f" % (
                        eval_results["test_thin"][target]["macro_f1"],
                    )
                logger.info(msg)

            if dry_run:
                logger.info(
                    "Dry run — skipping artifact saves and MLflow logging."
                )
                return {
                    "models": self.models,
                    "calibrated": self.calibrated,
                    "eval": eval_results,
                }

            # Save artifacts
            self.save_models()
            self.save_confusion_matrices(features)
            if mlflow_enabled(self.config):
                self._log_mlflow(eval_results)

        return {
            "models": self.models,
            "calibrated": self.calibrated,
            "eval": eval_results,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train and evaluate two-head LR baseline "
        "for vinyl condition grading"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate without saving artifacts",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow (mlflow.enabled: false)",
    )
    parser.add_argument(
        "--mlflow-no-artifacts",
        action="store_true",
        help="Params/metrics only (mlflow.log_artifacts: false). Ignored with --no-mlflow.",
    )
    args = parser.parse_args()

    cfg = load_yaml_mapping(args.config)
    ml = cfg.setdefault("mlflow", {})
    if args.no_mlflow:
        ml["enabled"] = False
    elif args.mlflow_no_artifacts and ml.get("enabled", True):
        ml["log_artifacts"] = False

    model = BaselineModel(config_path=args.config, config=cfg)
    model.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
