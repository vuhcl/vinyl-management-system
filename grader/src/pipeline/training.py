"""Full training orchestration (`Pipeline.train`)."""

from __future__ import annotations

import logging
from typing import Any

from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
)
from grader.src.pipeline_train_steps import (
    apply_train_mlflow_env_overrides,
    run_steps_1_through_4,
    run_train_steps_5_through_9,
)

logger = logging.getLogger(__name__)


class PipelineTrainingMixin:
    """Train subcommand: ingest through rule evaluation."""

    config: dict

    def train(
        self,
        skip_ingest: bool = False,
        skip_ebay_ingest: bool = False,
        skip_harmonize: bool = False,
        skip_preprocess: bool = False,
        skip_features: bool = False,
        skip_transformer: bool = False,
        baseline_only: bool = False,
        skip_baseline: bool = False,
        register_after_pipeline: bool | None = None,
        registry_model_name_override: str | None = None,
        no_mlflow: bool = False,
        mlflow_no_artifacts: bool = False,
        skip_sale_history_ingest: bool = False,
    ) -> dict[str, Any]:
        """
        Run the full training pipeline end to end.

        Steps:
          1. Ingest Discogs and eBay JP data
          2. Harmonize labels into unified dataset
          3. Preprocess text and split train/val/test
          4. Build TF-IDF features
          5. Train and evaluate baseline model (skippable via ``skip_baseline``)
          6. Train and evaluate transformer model (skippable)
          7. Compare baseline vs transformer metrics
          8. Generate calibration plots for both models
          9. Compute rule engine coverage on test split

        Args:
            skip_ingest:      skip steps 1 — use existing raw data
            skip_ebay_ingest: in step 1, run Discogs only (no eBay API / tokens)
            skip_harmonize:   skip step 2 — use existing unified.jsonl
            skip_preprocess:  skip step 3 — use existing split files
            skip_features:    skip step 4 — use existing feature matrices
            skip_transformer: skip step 6 — no DistilBERT training
            baseline_only:    alias for skip_transformer=True
            skip_baseline:     skip step 5 training; load baseline pickles from
                               paths.artifacts and evaluate on disk (Workflow A)
            skip_sale_history_ingest: if True, do not run sale-history
                               SQLite → discogs_sale_history.jsonl. Default is False;
                               use ``--skip-sale-history`` on ``pipeline train`` to opt out.
            register_after_pipeline: if None, use config ``mlflow.register_after_pipeline``
            registry_model_name_override: if set, overrides ``mlflow.registry_model_name``
            no_mlflow: if True, same as ``mlflow.enabled: false`` (no tracking).
            mlflow_no_artifacts: if True, params/metrics only; ignored if ``no_mlflow``.

        Returns:
            Dict with results from all completed steps.
        """
        skip_transformer = skip_transformer or baseline_only
        if skip_baseline and baseline_only:
            logger.warning(
                "Both skip_baseline and baseline_only: loading baseline from "
                "disk and skipping transformer training."
            )
        results: dict[str, Any] = {}

        ml = self.config.setdefault("mlflow", {})
        if no_mlflow:
            ml["enabled"] = False
            logger.info("MLflow disabled (--no-mlflow).")
        elif mlflow_no_artifacts and ml.get("enabled", True):
            ml["log_artifacts"] = False
            logger.info(
                "MLflow metrics-only (--mlflow-no-artifacts / "
                "mlflow.log_artifacts: false)."
            )
        apply_train_mlflow_env_overrides(self.config)

        if mlflow_enabled(self.config):
            configure_mlflow_from_config(self.config)

        mlflow_cfg = self.config.get("mlflow", {})
        if register_after_pipeline is None:
            want_registry = bool(mlflow_cfg.get("register_after_pipeline", True))
        else:
            want_registry = register_after_pipeline
        want_registry = (
            want_registry
            and mlflow_enabled(self.config)
            and mlflow_log_artifacts_enabled(self.config)
        )
        registry_model_name = (
            (registry_model_name_override or "").strip()
            or str(mlflow_cfg.get("registry_model_name", "VinylGrader"))
        )

        # Sub-steps (ingest/features/models/calibration) open their own MLflow runs.
        # Avoid opening an outer run here, otherwise nested start_run() calls fail.
        run_steps_1_through_4(
            self,
            results,
            skip_ingest=skip_ingest,
            skip_sale_history_ingest=skip_sale_history_ingest,
            skip_ebay_ingest=skip_ebay_ingest,
            skip_harmonize=skip_harmonize,
            skip_preprocess=skip_preprocess,
            skip_features=skip_features,
        )

        run_train_steps_5_through_9(
            self,
            results,
            skip_baseline=skip_baseline,
            skip_transformer=skip_transformer,
            want_registry=want_registry,
            registry_model_name=registry_model_name,
        )

        return results
