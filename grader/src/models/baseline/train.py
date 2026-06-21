"""Logistic-regression head training, orchestration, and auxiliary heads."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from grader.src.mlflow_tracking import (
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    mlflow_start_run_ctx,
)

from .constants import SPLITS, TARGETS

logger = logging.getLogger(__name__)


class _BaselineTrain:
    """Build and fit baseline heads, boundary blend, media aux, and full run."""

    config: dict
    C: float
    max_iter: int
    tol: float
    class_weight: str
    solver: str
    random_state: int
    tuning_enabled: bool
    tuning_c_values: list[float]
    boundary_enabled: bool
    boundary_alpha: float
    media_evidence_aux_enabled: bool
    calibration_method: str
    confidence_band_enabled: bool
    calibrated_paths: dict
    confusion_paths: dict
    models: dict
    calibrated: dict
    encoders: dict
    selected_c: dict[str, float]
    boundary_models: dict[str, LogisticRegression]
    boundary_band_labels: dict[str, np.ndarray]
    media_evidence_aux_model: Optional[LogisticRegression]

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

    def _log_mlflow(
        self,
        eval_results: dict[str, dict[str, dict]],
    ) -> None:
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

            features = self.load_all_features()
            self.encoders = self.load_encoders()

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

            self.calibrated = self.calibrate(self.models, features)

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

            self.save_models()
            self.save_confusion_matrices(features)
            if mlflow_enabled(self.config):
                self._log_mlflow(eval_results)

        return {
            "models": self.models,
            "calibrated": self.calibrated,
            "eval": eval_results,
        }
