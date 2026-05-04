"""
Training pipeline steps 1–4 (ingest → harmonize → preprocess → TF-IDF).

Extracted from :class:`grader.src.pipeline.Pipeline` to keep ``train()`` readable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import mlflow

from grader.src.data.harmonize_labels import LabelHarmonizer
from grader.src.data.ingest_discogs import DiscogsIngester
from grader.src.data.ingest_ebay import EbayIngester
from grader.src.data.ingest_sale_history import run_sale_history_ingest_from_config
from grader.src.data.label_patches import apply_label_patches_after_ingest
from grader.src.data.vinyl_format import run_post_patch_vinyl_filter_from_config
from grader.src.data.preprocess import Preprocessor
from grader.src.evaluation.calibration import CalibrationEvaluator
from grader.src.evaluation.metrics import (
    compare_models,
    compare_models_per_class,
    log_comparison_to_mlflow,
)
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.mlflow_tracking import mlflow_enabled, mlflow_log_artifacts_enabled
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer

logger = logging.getLogger(__name__)


def apply_train_mlflow_env_overrides(config: dict[str, Any]) -> None:
    """Apply ``VINYL_GRADER_MLFLOW_METRICS_ONLY`` env override to ``config``."""
    ml = config.setdefault("mlflow", {})
    if (
        os.environ.get("VINYL_GRADER_MLFLOW_METRICS_ONLY", "").strip().lower()
        in ("1", "true", "yes")
        and ml.get("enabled", True)
    ):
        ml["log_artifacts"] = False
        logger.info(
            "MLflow metrics-only (VINYL_GRADER_MLFLOW_METRICS_ONLY env set)."
        )


def run_steps_1_through_4(
    pipeline: Any,
    results: dict[str, Any],
    *,
    skip_ingest: bool,
    skip_sale_history_ingest: bool,
    skip_ebay_ingest: bool,
    skip_harmonize: bool,
    skip_preprocess: bool,
    skip_features: bool,
) -> None:
    """
    Steps 1–4: ingestion, harmonization, preprocessing, TF-IDF features.
    Mutates ``results`` in place.
    """
    # Step 1 — Ingestion
    if not skip_ingest:
        logger.info("=" * 50)
        logger.info("STEP 1 — DATA INGESTION")
        logger.info("=" * 50)

        repo_root = Path(__file__).resolve().parents[2]

        if not skip_sale_history_ingest:
            sh = run_sale_history_ingest_from_config(
                pipeline.config,
                Path(pipeline.config_path),
                Path(pipeline.guidelines_path),
                repo_root,
            )
            if sh.get("ok"):
                results["sale_history_ingest"] = sh
                trim = sh.get("trim") or {}
                sh_cfg = sh.get("sale_history_settings") or {}
                logger.info(
                    "Sale history → %s (%d line(s), vinyl post=%s, "
                    "trim in/joint/sleeve/total=%s/%s/%s/%s, prefetch=%s order=%s "
                    "joint_cap=%s sleeve_cap=%s total_cap=%s balance_sleeve_trim=%s)",
                    sh.get("out"),
                    sh.get("written", 0),
                    (sh.get("post") or {}).get("vinyl_dropped", 0),
                    trim.get("input_rows"),
                    trim.get("after_joint_trim"),
                    trim.get("after_sleeve_trim"),
                    trim.get("after_total_trim"),
                    sh_cfg.get("sql_prefetch_limit"),
                    sh_cfg.get("sql_sample_order"),
                    sh_cfg.get("max_rows_per_joint_grade"),
                    sh_cfg.get("max_rows_per_sleeve_grade"),
                    sh_cfg.get("max_total_sale_history_rows"),
                    sh_cfg.get("balance_joint_within_sleeve_trim"),
                )
            else:
                logger.warning(
                    "Sale history ingest not run: %s",
                    sh.get("error", sh),
                )

        discogs_ingester = DiscogsIngester(
            config_path=pipeline.config_path,
            guidelines_path=pipeline.guidelines_path,
            config=pipeline.config,
        )
        discogs_ingester.run()

        if skip_ebay_ingest:
            logger.info(
                "Skipping eBay ingestion (--skip-ebay-ingest) — "
                "harmonizer will use Discogs only if ebay_processed.jsonl "
                "is missing."
            )
        else:
            ebay_ingester = EbayIngester(
                config_path=pipeline.config_path,
                guidelines_path=pipeline.guidelines_path,
                config=pipeline.config,
            )
            ebay_ingester.run()

        patch_stats = apply_label_patches_after_ingest(pipeline.config)
        if patch_stats.get("enabled"):
            results["label_patches"] = patch_stats
            if patch_stats.get("updated_total", 0):
                logger.info(
                    "Label patches applied: %d row(s) updated "
                    "(see data.label_patches_path).",
                    patch_stats["updated_total"],
                )

        vinyl_post = run_post_patch_vinyl_filter_from_config(
            pipeline.config,
            filter_sale_jsonl=True,
        )
        if vinyl_post.get("ran"):
            results["discogs_vinyl_post_filter"] = vinyl_post
            logger.info(
                "Discogs post-patch vinyl filter: dropped=%s kept=%s",
                vinyl_post.get("dropped"),
                vinyl_post.get("kept"),
            )
    else:
        logger.info("Skipping ingestion — using existing raw data.")

    # Step 2 — Label harmonization
    if not skip_harmonize:
        logger.info("=" * 50)
        logger.info("STEP 2 — LABEL HARMONIZATION")
        logger.info("=" * 50)

        harmonizer = LabelHarmonizer(
            config_path=pipeline.config_path,
            guidelines_path=pipeline.guidelines_path,
            config=pipeline.config,
        )
        results["harmonize"] = harmonizer.run()
    else:
        logger.info("Skipping harmonization — using existing unified.jsonl.")

    # Step 3 — Preprocessing and splitting
    if not skip_preprocess:
        logger.info("=" * 50)
        logger.info("STEP 3 — PREPROCESSING AND SPLITTING")
        logger.info("=" * 50)

        preprocessor = Preprocessor(
            config_path=pipeline.config_path,
            guidelines_path=pipeline.guidelines_path,
            config=pipeline.config,
        )
        results["preprocess"] = preprocessor.run()
    else:
        logger.info("Skipping preprocessing — using existing splits.")

    # Step 4 — TF-IDF features
    if not skip_features:
        logger.info("=" * 50)
        logger.info("STEP 4 — TF-IDF FEATURE EXTRACTION")
        logger.info("=" * 50)

        tfidf = TFIDFFeatureBuilder(
            config_path=pipeline.config_path,
            config=pipeline.config,
        )
        results["features"] = tfidf.run()
    else:
        logger.info("Skipping feature extraction — using existing matrices.")


def run_train_steps_5_through_9(
    pipeline: Any,
    results: dict[str, Any],
    *,
    skip_baseline: bool,
    skip_transformer: bool,
    want_registry: bool,
    registry_model_name: str,
) -> None:
    """
    Steps 5–9: baseline, transformer, comparison, calibration, rule-engine eval.

    Mutates ``results`` in place. Uses :class:`Pipeline` helpers on ``pipeline``.
    """
    # Step 5 — Baseline model
    logger.info("=" * 50)
    logger.info("STEP 5 — BASELINE MODEL (TF-IDF + LR)")
    logger.info("=" * 50)

    if skip_baseline:
        logger.info(
            "Skipping baseline training — loading encoders + models "
            "from paths.artifacts and evaluating on disk features."
        )
        baseline, baseline_results = BaselineModel.load_trained_from_artifacts(
            pipeline.config_path
        )
        results["baseline"] = baseline_results
    else:
        baseline = BaselineModel(
            config_path=pipeline.config_path,
            config=pipeline.config,
        )
        baseline_results = baseline.run()
        results["baseline"] = baseline_results

    baseline_test_metrics = {
        target: baseline_results["eval"]["test"][target]
        for target in ["sleeve", "media"]
    }

    # Step 6 — Transformer model
    transformer_test_metrics = None
    trainer: Optional[TransformerTrainer] = None
    transformer_results: dict[str, Any] | None = None

    if not skip_transformer:
        logger.info("=" * 50)
        logger.info("STEP 6 — TRANSFORMER MODEL (DistilBERT)")
        logger.info("=" * 50)

        trainer = TransformerTrainer(
            config_path=pipeline.config_path,
            config=pipeline.config,
        )
        transformer_results = trainer.run()
        results["transformer"] = transformer_results

        transformer_test_metrics = {
            target: transformer_results["eval"]["test"][target]
            for target in ["sleeve", "media"]
        }
        pipeline._register_transformer_to_registry(
            transformer_results,
            want_register=want_registry,
            registry_model_name=registry_model_name,
        )
    else:
        logger.info("Skipping transformer — baseline only mode.")

    # Step 7 — Model comparison
    if transformer_test_metrics is not None and transformer_results is not None:
        logger.info("=" * 50)
        logger.info("STEP 7 — MODEL COMPARISON")
        logger.info("=" * 50)

        comparison_table = compare_models(
            baseline_metrics=baseline_test_metrics,
            transformer_metrics=transformer_test_metrics,
            split="test",
        )
        per_class_table = compare_models_per_class(
            baseline_metrics=baseline_test_metrics,
            transformer_metrics=transformer_test_metrics,
        )

        thin_compare_table = None
        thin_per_class_table = None
        if (
            "test_thin" in baseline_results["eval"]
            and "test_thin" in transformer_results["eval"]
        ):
            b_thin = {
                t: baseline_results["eval"]["test_thin"][t]
                for t in ["sleeve", "media"]
            }
            t_thin = {
                t: transformer_results["eval"]["test_thin"][t]
                for t in ["sleeve", "media"]
            }
            thin_compare_table = compare_models(
                baseline_metrics=b_thin,
                transformer_metrics=t_thin,
                split="test_thin",
            )
            thin_per_class_table = compare_models_per_class(
                baseline_metrics=b_thin,
                transformer_metrics=t_thin,
                split_title="TEST_THIN SPLIT",
            )

        print("\n" + comparison_table)
        print(per_class_table)
        if thin_compare_table is not None:
            print("\n" + thin_compare_table)
            print(thin_per_class_table)

        comparison_path = pipeline.artifacts_dir / "model_comparison.txt"
        with open(comparison_path, "w") as f:
            f.write(comparison_table + "\n\n" + per_class_table)
            if thin_compare_table is not None:
                f.write(
                    "\n\n"
                    + thin_compare_table
                    + "\n\n"
                    + thin_per_class_table
                )

        if mlflow_enabled(pipeline.config):
            if mlflow_log_artifacts_enabled(pipeline.config):
                mlflow.log_artifact(str(comparison_path))
            log_comparison_to_mlflow(
                baseline_metrics=baseline_test_metrics,
                transformer_metrics=transformer_test_metrics,
            )
            if thin_compare_table is not None:
                log_comparison_to_mlflow(
                    baseline_metrics=b_thin,
                    transformer_metrics=t_thin,
                    key_suffix="test_thin",
                )

        results["comparison"] = {
            "table": comparison_table,
            "per_class": per_class_table,
        }
        if thin_compare_table is not None:
            results["comparison"]["test_thin_table"] = thin_compare_table
            results["comparison"]["test_thin_per_class"] = thin_per_class_table

    # Step 8 — Calibration plots
    logger.info("=" * 50)
    logger.info("STEP 8 — CALIBRATION EVALUATION")
    logger.info("=" * 50)

    calibration_evaluator = CalibrationEvaluator(
        config_path=pipeline.config_path
    )

    pipeline._get_tfidf()
    _cal_mlf = (
        mlflow_enabled(pipeline.config)
        and mlflow_log_artifacts_enabled(pipeline.config)
    )
    for target in ["sleeve", "media"]:
        X_test, y_test = TFIDFFeatureBuilder.load_features(
            str(pipeline.artifacts_dir / "features"),
            split="test",
            target=target,
        )
        encoder = TFIDFFeatureBuilder.load_encoder(
            str(pipeline.artifacts_dir / f"label_encoder_{target}.pkl")
        )

        baseline_proba = BaselineModel.load_model(
            str(pipeline.artifacts_dir / f"baseline_{target}_calibrated.pkl")
        ).predict_proba(X_test)

        calibration_evaluator.run(
            y_true=y_test,
            y_proba=baseline_proba,
            class_names=encoder.classes_,
            target=target,
            model_name="baseline",
            log_to_mlflow=_cal_mlf,
        )

        if transformer_test_metrics is not None and transformer_results is not None:
            transformer_proba = transformer_results["eval"]["test"][target][
                "y_proba"
            ]
            calibration_evaluator.run(
                y_true=y_test,
                y_proba=transformer_proba,
                class_names=encoder.classes_,
                target=target,
                model_name="transformer",
                log_to_mlflow=_cal_mlf,
            )

    # Step 9 — Rule engine: compare model vs adjusted; audit overrides
    logger.info("=" * 50)
    _ev = pipeline.config.get("evaluation") or {}
    _rs = _ev.get("rule_eval_splits")
    _split_note = (
        ", ".join(str(s) for s in _rs)
        if isinstance(_rs, list) and _rs
        else "test + test_thin"
    )
    logger.info("STEP 9 — RULE ENGINE EVAL (%s)", _split_note)
    logger.info("=" * 50)

    rule_engine = pipeline._get_rule_engine()
    use_transformer = transformer_test_metrics is not None

    rule_eval, rule_mlflow, grade_paths = pipeline._run_rule_engine_evaluation(
        rule_engine=rule_engine,
        trainer=trainer if use_transformer else None,
        baseline=baseline,
        use_transformer=use_transformer,
    )

    if mlflow_enabled(pipeline.config):
        mlflow.log_metrics(rule_mlflow)
        if mlflow_log_artifacts_enabled(pipeline.config):
            for _p in grade_paths.values():
                mlflow.log_artifact(_p)

    results["rule_eval"] = rule_eval
    results["grade_analysis_reports"] = grade_paths
    if "test" in rule_eval:
        results["rule_adjusted_test_metrics"] = rule_eval["test"]["adjusted"]
        results["rule_coverage"] = rule_eval["test"]["coverage"]

    logger.info("=" * 50)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 50)
