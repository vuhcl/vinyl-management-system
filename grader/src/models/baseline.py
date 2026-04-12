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

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.mlflow_tracking import configure_mlflow_from_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TARGETS = ["sleeve", "media"]
SPLITS = ["train", "val", "test"]


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

    def __init__(self, config_path: str) -> None:
        self.config = self._load_yaml(config_path)

        lr_cfg = self.config["models"]["baseline"]["logistic_regression"]
        self.C: float = lr_cfg["C"]
        self.max_iter: int = lr_cfg["max_iter"]
        self.class_weight: str = lr_cfg["class_weight"]
        self.solver: str = lr_cfg["solver"]
        self.random_state: int = lr_cfg.get("random_state", 42)

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
        self.calibrated: dict[str, CalibratedClassifierCV] = {}
        self.encoders: dict = {}

        # MLflow — resolve tracking URI (env / fallback / legacy key)
        configure_mlflow_from_config(self.config)

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

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
    def build_model(self, target: str) -> LogisticRegression:
        """
        Construct a LogisticRegression head for a single target.
        Hyperparameters come from grader.yaml — nothing hardcoded.
        """
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )

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

            model = self.build_model(target)
            model.fit(X_train, y_train)
            models[target] = model

            train_preds = model.predict(X_train)
            train_f1 = f1_score(y_train, train_preds, average="macro")
            logger.info("Train macro-F1 — target=%s: %.4f", target, train_f1)

        return models

    # -----------------------------------------------------------------------
    # Calibration — fitted on val split
    # -----------------------------------------------------------------------
    def calibrate(
        self,
        models: dict[str, LogisticRegression],
        features: dict,
    ) -> dict[str, CalibratedClassifierCV]:
        """
        Wrap each fitted model in CalibratedClassifierCV and fit
        calibration on the val split.

        Calibration is fitted on val — NOT train — to prevent
        calibration from absorbing training signal.

        Args:
            models:   fitted LogisticRegression heads
            features: nested dict from load_all_features()

        Returns:
            Dict mapping target → calibrated classifier.
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

            calibrator = CalibratedClassifierCV(
                estimator=models[target],
                method=self.calibration_method,
            )
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
        sleeve_pred_idx = self.calibrated["sleeve"].predict(X_sleeve)
        sleeve_proba = self.calibrated["sleeve"].predict_proba(X_sleeve)
        media_pred_idx = self.calibrated["media"].predict(X_media)
        media_proba = self.calibrated["media"].predict_proba(X_media)

        # Decode integer indices back to grade strings
        sleeve_pred = sleeve_encoder.inverse_transform(sleeve_pred_idx)
        media_pred = media_encoder.inverse_transform(media_pred_idx)

        sleeve_classes = sleeve_encoder.classes_
        media_classes = media_encoder.classes_

        predictions = []
        for i in range(n_samples):
            # Build confidence score dicts
            sleeve_scores = {
                cls: round(float(sleeve_proba[i][j]), 4)
                for j, cls in enumerate(sleeve_classes)
            }
            media_scores = {
                cls: round(float(media_proba[i][j]), 4)
                for j, cls in enumerate(media_classes)
            }

            # Extract metadata from source record if available
            record = records[i] if records else {}
            media_verifiable = record.get("media_verifiable", True)
            contradiction = record.get("contradiction_detected", False)
            source = record.get("source", "unknown")

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
            split:    "train", "val", or "test"

        Returns:
            Dict mapping target → metrics dict.
        """
        results = {}

        for target in TARGETS:
            X = features[split][target]["X"]
            y_true = features[split][target]["y"]
            encoder = self.encoders[target]

            y_pred = self.calibrated[target].predict(X)
            y_proba = self.calibrated[target].predict_proba(X)

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
        cm = confusion_matrix(y_true, y_pred)

        col_width = max(len(c) for c in classes) + 2
        header = " " * col_width + "".join(f"{c:>{col_width}}" for c in classes)
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
            y_pred = self.calibrated[target].predict(X)

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
                "lr_class_weight": self.class_weight,
                "lr_solver": self.solver,
                "calibration_method": self.calibration_method,
            }
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
                # Per-class F1 on test split
                if split == "test":
                    for class_name, class_metrics in metrics["report"].items():
                        if isinstance(class_metrics, dict):
                            clean = class_name.lower().replace(" ", "_")
                            mlflow.log_metric(
                                f"test_{target}_{clean}_f1",
                                class_metrics.get("f1-score", 0.0),
                            )

        # Artifacts
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
        with mlflow.start_run(run_name="baseline_tfidf_logreg"):

            # Load features and encoders
            features = self.load_all_features()
            self.encoders = self.load_encoders()

            # Train
            self.models = self.train(features)

            # Calibrate on val
            self.calibrated = self.calibrate(self.models, features)

            # Evaluate on all splits
            eval_results: dict[str, dict] = {}
            for split in SPLITS:
                logger.info("--- Evaluating on %s split ---", split.upper())
                eval_results[split] = self.evaluate(features, split)

            # Summary log
            for target in TARGETS:
                logger.info(
                    "RESULTS SUMMARY — target=%s | "
                    "train macro-F1: %.4f | "
                    "val macro-F1: %.4f | "
                    "test macro-F1: %.4f",
                    target,
                    eval_results["train"][target]["macro_f1"],
                    eval_results["val"][target]["macro_f1"],
                    eval_results["test"][target]["macro_f1"],
                )

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
    args = parser.parse_args()

    model = BaselineModel(config_path=args.config)
    model.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
