"""Full training run wiring and MLflow logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

from grader.src.mlflow_tracking import (
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    mlflow_start_run_ctx,
)

from .constants import SPLITS, TARGETS

logger = logging.getLogger(__name__)


class BaselineOrchestrationMixin:
    """End-to-end baseline train/eval/save and MLflow."""

    config: dict
    calibrated_paths: dict
    confusion_paths: dict
    selected_c: dict[str, float]
    C: float
    max_iter: int
    tol: float
    class_weight: str
    solver: str
    calibration_method: str
    tuning_enabled: bool
    boundary_enabled: bool
    boundary_alpha: float
    confidence_band_enabled: bool
    media_gating_enabled: bool
    media_gating_alpha_none: float
    media_gating_alpha_weak: float
    media_evidence_aux_enabled: bool

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
