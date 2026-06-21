"""Full training orchestration and rule-engine evaluation helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
)
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer
from grader.src.pipeline_train_steps import (
    apply_train_mlflow_env_overrides,
    run_steps_1_through_4,
    run_train_steps_5_through_9,
)
from grader.src.rules.rule_engine import RuleEngine
from grader.src.schemas import GraderPrediction, merge_description_quality_metadata

logger = logging.getLogger(__name__)


class _PipelineTraining:
    """Train subcommand: ingest through rule evaluation."""

    config: dict
    artifacts_dir: Path

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

    def _baseline_predict_from_features(
        self,
        baseline: BaselineModel,
        split: str = "test",
    ) -> list[GraderPrediction]:
        """
        Rebuild baseline predictions from saved TF-IDF features for a split.
        Used when the transformer is not run (rule eval uses the same path).
        """
        X_sleeve, _ = TFIDFFeatureBuilder.load_features(
            str(self.artifacts_dir / "features"),
            split=split,
            target="sleeve",
        )
        X_media, _ = TFIDFFeatureBuilder.load_features(
            str(self.artifacts_dir / "features"),
            split=split,
            target="media",
        )
        split_records = self._load_split(split)
        item_ids = [
            r.get("item_id", str(i)) for i, r in enumerate(split_records)
        ]

        return baseline.predict(
            X_sleeve=X_sleeve,
            X_media=X_media,
            item_ids=item_ids,
            records=split_records,
        )

    def _predictions_for_rule_eval(
        self,
        split: str,
        trainer: Optional[TransformerTrainer],
        baseline: BaselineModel,
        use_transformer: bool,
    ) -> tuple[list[GraderPrediction], list[str]]:
        """Model-only predictions and aligned text for rule evaluation."""
        records = self._load_split(split)
        texts = [r.get("text_clean") or r.get("text", "") for r in records]
        item_ids = [r.get("item_id") for r in records]
        if use_transformer:
            if trainer is None:
                raise RuntimeError("use_transformer=True but trainer is None")
            preds = trainer.predict(
                texts=texts,
                item_ids=item_ids,
                records=records,
            )
        else:
            preds = self._baseline_predict_from_features(baseline, split=split)
        merge_description_quality_metadata(preds, records)
        return preds, texts

    def _run_rule_engine_evaluation(
        self,
        rule_engine: RuleEngine,
        trainer: Optional[TransformerTrainer],
        baseline: BaselineModel,
        use_transformer: bool,
    ) -> tuple[dict, dict[str, float], dict[str, str]]:
        """
        Rule-adjusted metrics, model-only metrics, and override audit on
        test and test_thin (when split + features exist).

        Also writes ``grade_analysis_{split}.txt`` (including a
        ``RULE-OWNED SLICE`` banner + stratified override-audit section)
        and a dual-format baseline snapshot to
        ``rule_engine_baseline.json`` plus MLflow tags keyed
        ``rule_baseline_*`` — see §8 of the rule-owned eval plan.
        """
        from grader.src.evaluation.rule_engine_eval import (
            run_rule_engine_evaluation,
        )

        return run_rule_engine_evaluation(
            self,
            rule_engine,
            trainer,
            baseline,
            use_transformer,
        )

    def _load_split(self, split: str) -> list[dict]:
        """Load records from a split JSONL file."""
        splits_dir = Path(self.config["paths"]["splits"])
        path = splits_dir / f"{split}.jsonl"
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
