"""
grader/src/pipeline.py

Top-level orchestrator for the vinyl condition grader.
Exposes two distinct pipelines:

  1. Training pipeline  — end-to-end: ingest → preprocess →
                          features → baseline → transformer →
                          compare → calibration → rule coverage

  2. Inference pipeline — single or batch: raw text →
                          preprocess → model predict →
                          rule engine → final prediction

All preprocessing is handled internally at inference time.
Callers (iOS app, CLI, batch job) pass raw text only.

Usage:
    # Training
    python -m grader.src.pipeline train
    python -m grader.src.pipeline train --skip-ingest
    python -m grader.src.pipeline train --skip-sale-history  # omit sale_history → JSONL
    python -m grader.src.pipeline train --baseline-only

    # Inference
    python -m grader.src.pipeline predict --text "NM sleeve, plays perfectly"
    python -m grader.src.pipeline predict --file texts.txt
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import yaml

from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    vinyl_grader_pyfunc_has_python_model,
)
from grader.src.data.harmonize_labels import LabelHarmonizer
from grader.src.data.ingest_discogs import DiscogsIngester
from grader.src.data.ingest_ebay import EbayIngester
from grader.src.data.label_patches import apply_label_patches_after_ingest
from grader.src.data.ingest_sale_history import run_sale_history_ingest_from_config
from grader.src.data.vinyl_format import run_post_patch_vinyl_filter_from_config
from grader.src.data.preprocess import Preprocessor
from grader.src.evaluation.calibration import CalibrationEvaluator
from grader.src.evaluation.grade_analysis import (
    build_grade_analysis_report,
    build_rule_owned_slice_report,
    resolve_rule_owned_grades,
    slice_recall_for_grade,
)
from grader.src.evaluation.metrics import (
    compare_models,
    compare_models_per_class,
    compute_metrics_from_label_strings,
    compute_rule_override_audit,
    format_override_audit_report,
    log_comparison_to_mlflow,
    remap_true_and_encode_predictions,
    substitute_model_when_pred_excellent,
)
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer
from grader.src.rules.rule_engine import RuleEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    """
    Top-level orchestrator for training and inference.

    Training pipeline:
        Runs the full sequence from ingestion to model comparison.
        Each step is individually skippable for development iterations.

    Inference pipeline:
        Loads pre-trained artifacts and runs single or batch prediction.
        All preprocessing is handled internally — callers pass raw text.

    Config keys read from grader.yaml:
        inference.model             — "baseline" or "transformer"
        paths.*                     — all artifact and data paths
        mlflow.*                    — experiment tracking config
    """

    def __init__(
        self,
        config_path: str = "grader/configs/grader.yaml",
        guidelines_path: str = "grader/configs/grading_guidelines.yaml",
    ) -> None:
        self.config_path     = config_path
        self.guidelines_path = guidelines_path
        self.config          = self._load_yaml(config_path)

        # Inference model selection
        inference_cfg    = self.config.get("inference", {})
        self.infer_model = inference_cfg.get("model", "transformer")

        # Paths
        self.artifacts_dir = Path(self.config["paths"]["artifacts"])

        if mlflow_enabled(self.config):
            configure_mlflow_from_config(self.config)

        # Lazy-loaded inference components — initialized on first predict call
        self._preprocessor: Optional[Preprocessor]        = None
        self._rule_engine:  Optional[RuleEngine]           = None
        self._baseline:     Optional[BaselineModel]        = None
        self._transformer:  Optional[TransformerTrainer]   = None
        self._tfidf:        Optional[TFIDFFeatureBuilder]  = None

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Lazy component initialization
    # -----------------------------------------------------------------------
    def _get_preprocessor(self) -> Preprocessor:
        if self._preprocessor is None:
            self._preprocessor = Preprocessor(
                config_path=self.config_path,
                guidelines_path=self.guidelines_path,
            )
        return self._preprocessor

    def _get_rule_engine(self) -> RuleEngine:
        if self._rule_engine is None:
            rules_cfg = self.config.get("rules") or {}
            allow_ex = bool(
                rules_cfg.get("allow_excellent_soft_override", False)
            )
            self._rule_engine = RuleEngine(
                guidelines_path=self.guidelines_path,
                allow_excellent_soft_override=allow_ex,
            )
        return self._rule_engine

    def _get_tfidf(self) -> TFIDFFeatureBuilder:
        if self._tfidf is None:
            self._tfidf = TFIDFFeatureBuilder(
                config_path=self.config_path
            )
        return self._tfidf

    # -----------------------------------------------------------------------
    # MLflow model registry (full pipeline)
    # -----------------------------------------------------------------------
    def _register_transformer_to_registry(
        self,
        transformer_results: dict,
        *,
        want_register: bool,
        registry_model_name: str,
    ) -> None:
        if not want_register:
            return
        if not mlflow_log_artifacts_enabled(self.config):
            logger.info(
                "Skipping MLflow model registry (mlflow.log_artifacts: false)."
            )
            return
        run_id = (transformer_results or {}).get("mlflow_run_id") or ""
        if not run_id:
            logger.warning(
                "Skipping MLflow model registry: empty mlflow_run_id "
                "(enable MLflow for the transformer step to register)."
            )
            return
        configure_mlflow_from_config(self.config)
        if not vinyl_grader_pyfunc_has_python_model(run_id):
            logger.warning(
                "Skipping MLflow model registry: run %s has no usable "
                "vinyl_grader pyfunc (no python_model.pkl and no models-from-code "
                "entry in MLmodel). Remote pyfunc logging may have failed "
                "(see training logs for 'pyfunc logging failed'); increase "
                "MLFLOW_HTTP_REQUEST_TIMEOUT and related upload settings — "
                "grader/serving/README.md.",
                run_id,
            )
            return
        model_uri = f"runs:/{run_id}/vinyl_grader"
        try:
            ev = transformer_results.get("eval", {}).get("test", {})
            s_f1 = float(ev.get("sleeve", {}).get("macro_f1", 0.0))
            m_f1 = float(ev.get("media", {}).get("macro_f1", 0.0))
            mean_f1 = (s_f1 + m_f1) / 2.0
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=registry_model_name,
                tags={
                    "source": "full_pipeline",
                    "test_mean_macro_f1": str(round(mean_f1, 4)),
                },
            )
            logger.info(
                "Registered model '%s' version %s from run %s (%s)",
                registry_model_name,
                mv.version,
                run_id,
                model_uri,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model registration failed: %s", exc)

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
    ) -> dict:
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
        results = {}

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
        if (
            os.environ.get("VINYL_GRADER_MLFLOW_METRICS_ONLY", "").strip().lower()
            in ("1", "true", "yes")
            and ml.get("enabled", True)
        ):
            ml["log_artifacts"] = False
            logger.info(
                "MLflow metrics-only (VINYL_GRADER_MLFLOW_METRICS_ONLY env set)."
            )

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
        if True:

            # Step 1 — Ingestion
            if not skip_ingest:
                logger.info("=" * 50)
                logger.info("STEP 1 — DATA INGESTION")
                logger.info("=" * 50)

                discogs_ingester = DiscogsIngester(
                    config_path=self.config_path,
                    guidelines_path=self.guidelines_path,
                    config=self.config,
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
                        config_path=self.config_path,
                        guidelines_path=self.guidelines_path,
                        config=self.config,
                    )
                    ebay_ingester.run()

                patch_stats = apply_label_patches_after_ingest(self.config)
                if patch_stats.get("enabled"):
                    results["label_patches"] = patch_stats
                    if patch_stats.get("updated_total", 0):
                        logger.info(
                            "Label patches applied: %d row(s) updated "
                            "(see data.label_patches_path).",
                            patch_stats["updated_total"],
                        )

                vinyl_post = run_post_patch_vinyl_filter_from_config(
                    self.config,
                    filter_sale_jsonl=bool(skip_sale_history_ingest),
                )
                if vinyl_post.get("ran"):
                    results["discogs_vinyl_post_filter"] = vinyl_post
                    logger.info(
                        "Discogs post-patch vinyl filter: dropped=%s kept=%s",
                        vinyl_post.get("dropped"),
                        vinyl_post.get("kept"),
                    )
                if not skip_sale_history_ingest:
                    repo_root = Path(__file__).resolve().parents[2]
                    sh = run_sale_history_ingest_from_config(
                        self.config,
                        Path(self.config_path),
                        Path(self.guidelines_path),
                        repo_root,
                    )
                    if sh.get("ok"):
                        results["sale_history_ingest"] = sh
                        logger.info(
                            "Sale history → %s (%d line(s), vinyl drop in post: %s)",
                            sh.get("out"),
                            sh.get("written", 0),
                            (sh.get("post") or {}).get("vinyl_dropped", 0),
                        )
                    else:
                        logger.warning(
                            "Sale history ingest not run: %s",
                            sh.get("error", sh),
                        )
            else:
                logger.info("Skipping ingestion — using existing raw data.")
            # Step 2 — Label harmonization
            if not skip_harmonize:
                logger.info("=" * 50)
                logger.info("STEP 2 — LABEL HARMONIZATION")
                logger.info("=" * 50)

                harmonizer = LabelHarmonizer(
                    config_path=self.config_path,
                    guidelines_path=self.guidelines_path,
                    config=self.config,
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
                    config_path=self.config_path,
                    guidelines_path=self.guidelines_path,
                    config=self.config,
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
                    config_path=self.config_path,
                    config=self.config,
                )
                results["features"] = tfidf.run()
            else:
                logger.info("Skipping feature extraction — using existing matrices.")

            # Step 5 — Baseline model
            logger.info("=" * 50)
            logger.info("STEP 5 — BASELINE MODEL (TF-IDF + LR)")
            logger.info("=" * 50)

            if skip_baseline:
                logger.info(
                    "Skipping baseline training — loading encoders + models "
                    "from paths.artifacts and evaluating on disk features."
                )
                baseline, baseline_results = (
                    BaselineModel.load_trained_from_artifacts(self.config_path)
                )
                results["baseline"] = baseline_results
            else:
                baseline = BaselineModel(
                    config_path=self.config_path,
                    config=self.config,
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

            if not skip_transformer:
                logger.info("=" * 50)
                logger.info("STEP 6 — TRANSFORMER MODEL (DistilBERT)")
                logger.info("=" * 50)

                trainer = TransformerTrainer(
                    config_path=self.config_path,
                    config=self.config,
                )
                transformer_results = trainer.run()
                results["transformer"] = transformer_results

                transformer_test_metrics = {
                    target: transformer_results["eval"]["test"][target]
                    for target in ["sleeve", "media"]
                }
                self._register_transformer_to_registry(
                    transformer_results,
                    want_register=want_registry,
                    registry_model_name=registry_model_name,
                )
            else:
                logger.info(
                    "Skipping transformer — baseline only mode."
                )

            # Step 7 — Model comparison
            if transformer_test_metrics is not None:
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

                # Save comparison tables as artifacts
                comparison_path = (
                    self.artifacts_dir / "model_comparison.txt"
                )
                with open(comparison_path, "w") as f:
                    f.write(comparison_table + "\n\n" + per_class_table)
                    if thin_compare_table is not None:
                        f.write(
                            "\n\n"
                            + thin_compare_table
                            + "\n\n"
                            + thin_per_class_table
                        )

                if mlflow_enabled(self.config):
                    if mlflow_log_artifacts_enabled(self.config):
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
                    "table":     comparison_table,
                    "per_class": per_class_table,
                }
                if thin_compare_table is not None:
                    results["comparison"]["test_thin_table"] = (
                        thin_compare_table
                    )
                    results["comparison"]["test_thin_per_class"] = (
                        thin_per_class_table
                    )

            # Step 8 — Calibration plots
            logger.info("=" * 50)
            logger.info("STEP 8 — CALIBRATION EVALUATION")
            logger.info("=" * 50)

            calibration_evaluator = CalibrationEvaluator(
                config_path=self.config_path
            )

            # Load test features for calibration plots
            tfidf_builder = self._get_tfidf()
            _cal_mlf = (
                mlflow_enabled(self.config)
                and mlflow_log_artifacts_enabled(self.config)
            )
            for target in ["sleeve", "media"]:
                X_test, y_test = TFIDFFeatureBuilder.load_features(
                    str(self.artifacts_dir / "features"),
                    split="test",
                    target=target,
                )
                encoder = TFIDFFeatureBuilder.load_encoder(
                    str(self.artifacts_dir / f"label_encoder_{target}.pkl")
                )

                # Baseline calibration
                baseline_proba = BaselineModel.load_model(
                    str(self.artifacts_dir / f"baseline_{target}_calibrated.pkl")
                ).predict_proba(X_test)

                calibration_evaluator.run(
                    y_true=y_test,
                    y_proba=baseline_proba,
                    class_names=encoder.classes_,
                    target=target,
                    model_name="baseline",
                    log_to_mlflow=_cal_mlf,
                )

                # Transformer calibration — if available
                if transformer_test_metrics is not None:
                    transformer_proba = (
                        transformer_results["eval"]["test"][target]["y_proba"]
                    )
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
            logger.info("STEP 9 — RULE ENGINE EVAL (test + test_thin)")
            logger.info("=" * 50)

            rule_engine = self._get_rule_engine()
            use_transformer = transformer_test_metrics is not None

            rule_eval, rule_mlflow, grade_paths = (
                self._run_rule_engine_evaluation(
                    rule_engine=rule_engine,
                    trainer=trainer if use_transformer else None,
                    baseline=baseline,
                    use_transformer=use_transformer,
                )
            )

            if mlflow_enabled(self.config):
                mlflow.log_metrics(rule_mlflow)
                if mlflow_log_artifacts_enabled(self.config):
                    for _p in grade_paths.values():
                        mlflow.log_artifact(_p)

            results["rule_eval"] = rule_eval
            results["grade_analysis_reports"] = grade_paths
            if "test" in rule_eval:
                results["rule_adjusted_test_metrics"] = rule_eval["test"][
                    "adjusted"
                ]
                results["rule_coverage"] = rule_eval["test"]["coverage"]

            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE COMPLETE")
            logger.info("=" * 50)

        return results

    # -----------------------------------------------------------------------
    # Inference pipeline — single prediction
    # -----------------------------------------------------------------------
    def predict(
        self,
        text: str,
        item_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Run inference on a single raw text input.

        Handles all preprocessing internally. Callers pass raw text
        (voice-transcribed or typed) and receive a structured prediction.

        Args:
            text:     raw seller notes or user description
            item_id:  optional item identifier for output
            metadata: optional dict with additional context
                      (source, media_verifiable, etc.)

        Returns:
            Final prediction dict with rule engine applied.
        """
        results = self.predict_batch(
            texts=[text],
            item_ids=[item_id] if item_id else None,
            metadata_list=[metadata] if metadata else None,
        )
        return results[0]

    # -----------------------------------------------------------------------
    # Inference pipeline — batch prediction
    # -----------------------------------------------------------------------
    def predict_batch(
        self,
        texts: list[str],
        item_ids: Optional[list[str]] = None,
        metadata_list: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Run inference on a batch of raw text inputs.

        Steps:
          1. Preprocess each text (normalize, expand abbreviations)
          2. Run model prediction (baseline or transformer)
          3. Apply rule engine post-processing
          4. Return final predictions

        Args:
            texts:         list of raw text strings
            item_ids:      optional list of item IDs
            metadata_list: optional list of metadata dicts per item

        Returns:
            List of final prediction dicts with rule engine applied.
        """
        if not texts:
            return []

        if item_ids is None:
            item_ids = [str(i) for i in range(len(texts))]

        if metadata_list is None:
            metadata_list = [{} for _ in texts]

        preprocessor = self._get_preprocessor()
        rule_engine  = self._get_rule_engine()

        # Step 1 — Preprocess raw text
        # Run detection on raw text first (required by preprocess.py design)
        # then clean for model input
        clean_texts = []
        records     = []

        for i, (text, meta) in enumerate(zip(texts, metadata_list)):
            media_verifiable = preprocessor.detect_unverified_media(text)
            media_evidence_strength = preprocessor.detect_media_evidence_strength(
                text
            )
            text_clean       = preprocessor.clean_text(text)
            clean_texts.append(text_clean)

            record = {
                "item_id":         item_ids[i],
                "text":            text,
                "text_clean":      text_clean,
                "media_verifiable": media_verifiable,
                "media_evidence_strength": media_evidence_strength,
                "source":          meta.get("source", "user_input"),
                **meta,
            }
            record.update(
                preprocessor.compute_description_quality(text, text_clean)
            )
            records.append(record)

        # Step 2 — Model prediction
        predictions = self._model_predict(
            clean_texts=clean_texts,
            item_ids=item_ids,
            records=records,
        )
        self._merge_description_metadata(predictions, records)

        # Step 3 — Rule engine post-processing
        final_predictions = rule_engine.apply_batch(
            predictions=predictions,
            texts=clean_texts,
        )

        return final_predictions

    @staticmethod
    def _merge_description_metadata(
        predictions: list[dict],
        records: list[dict],
    ) -> None:
        """Copy note-adequacy fields from preprocess records into model metadata."""
        keys = (
            "sleeve_note_adequate",
            "media_note_adequate",
            "adequate_for_training",
            "needs_richer_note",
            "description_quality_gaps",
            "description_quality_prompts",
        )
        for pred, rec in zip(predictions, records):
            meta = pred.setdefault("metadata", {})
            for k in keys:
                if k in rec:
                    meta[k] = rec[k]

    # -----------------------------------------------------------------------
    # Model prediction dispatch
    # -----------------------------------------------------------------------
    def _model_predict(
        self,
        clean_texts: list[str],
        item_ids: list[str],
        records: list[dict],
    ) -> list[dict]:
        """
        Dispatch prediction to the configured model (baseline or transformer).
        Loads model artifacts lazily on first call.
        """
        if self.infer_model == "transformer":
            return self._transformer_predict(clean_texts, item_ids, records)
        else:
            return self._baseline_predict(clean_texts, item_ids, records)

    def _transformer_predict(
        self,
        clean_texts: list[str],
        item_ids: list[str],
        records: list[dict],
    ) -> list[dict]:
        """Load transformer artifacts and run prediction."""
        if self._transformer is None:
            logger.info("Loading transformer model from artifacts ...")
            self._transformer = TransformerTrainer(
                config_path=self.config_path
            )
            self._transformer.encoders = (
                self._transformer.load_encoders()
            )
            self._transformer.load_model()

        return self._transformer.predict(
            texts=clean_texts,
            item_ids=item_ids,
            records=records,
        )

    def _baseline_predict(
        self,
        clean_texts: list[str],
        item_ids: list[str],
        records: list[dict],
    ) -> list[dict]:
        """
        Load baseline artifacts and run prediction.
        Vectorizes text using the fitted TF-IDF vectorizer.
        """
        if self._baseline is None:
            logger.info("Loading baseline model from artifacts ...")
            self._baseline = BaselineModel(config_path=self.config_path)
            self._baseline.encoders = self._baseline.load_encoders()

            for target in ["sleeve", "media"]:
                cal_path = (
                    self.artifacts_dir
                    / f"baseline_{target}_calibrated.pkl"
                )
                self._baseline.calibrated[target] = (
                    BaselineModel.load_model(str(cal_path))
                )

        tfidf = self._get_tfidf()

        # Vectorize texts (TF-IDF + optional engineered features)
        sleeve_vectorizer = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_sleeve.pkl")
        )
        media_vectorizer = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_media.pkl")
        )
        X_sleeve = tfidf.transform_records(
            vectorizer=sleeve_vectorizer,
            records=records,
            target="sleeve",
            split="inference",
        )
        X_media = tfidf.transform_records(
            vectorizer=media_vectorizer,
            records=records,
            target="media",
            split="inference",
        )

        return self._baseline.predict(
            X_sleeve=X_sleeve,
            X_media=X_media,
            item_ids=item_ids,
            records=records,
        )

    def _baseline_predict_from_features(
        self,
        baseline: BaselineModel,
        split: str = "test",
    ) -> list[dict]:
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
        item_ids = [r.get("item_id", str(i)) for i, r in enumerate(split_records)]

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
    ) -> tuple[list[dict], list[str]]:
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
        self._merge_description_metadata(preds, records)
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
        features_dir = str(self.artifacts_dir / "features")
        splits_dir = Path(self.config["paths"]["splits"])
        out: dict = {}
        mlflow_flat: dict[str, float] = {}
        grade_analysis_paths: dict[str, str] = {}
        use_excellent_blend = bool(
            self.config.get("evaluation", {}).get(
                "excellent_eval_use_model_prediction", False
            )
        )
        rule_owned_grades = resolve_rule_owned_grades(rule_engine.guidelines)
        # Baseline snapshot accumulator: {split: {target: {...}}}
        baseline_snapshot: dict[str, dict] = {}

        for split_name in ("test", "test_thin"):
            if split_name == "test_thin" and not (splits_dir / "test_thin.jsonl").exists():
                logger.info(
                    "Rule eval — skip split=test_thin (no test_thin.jsonl)"
                )
                continue
            if split_name == "test_thin":
                try:
                    TFIDFFeatureBuilder.load_features(
                        features_dir, split="test_thin", target="sleeve"
                    )
                except OSError:
                    logger.info(
                        "Rule eval — skip split=test_thin (no feature matrices)"
                    )
                    continue

            raw, texts = self._predictions_for_rule_eval(
                split_name, trainer, baseline, use_transformer
            )
            adjusted = rule_engine.apply_batch(raw, texts)
            coverage = rule_engine.summarize_results(adjusted)

            adjusted_m: dict[str, dict] = {}
            adjusted_raw_m: dict[str, dict] = {}
            model_m: dict[str, dict] = {}
            audit_m: dict[str, dict] = {}
            grade_report_sections: list[str] = []

            for target in ("sleeve", "media"):
                _, y = TFIDFFeatureBuilder.load_features(
                    features_dir, split=split_name, target=target
                )
                encoder = TFIDFFeatureBuilder.load_encoder(
                    str(self.artifacts_dir / f"label_encoder_{target}.pkl")
                )
                pred_key = f"predicted_{target}_condition"
                before = [str(p[pred_key]) for p in raw]
                after = [str(p[pred_key]) for p in adjusted]
                after_eval = (
                    substitute_model_when_pred_excellent(after, before)
                    if use_excellent_blend
                    else after
                )
                n_ex_subst = (
                    sum(1 for a, e in zip(after, after_eval) if a != e)
                    if use_excellent_blend
                    else 0
                )
                adjusted_m[target] = compute_metrics_from_label_strings(
                    y,
                    after_eval,
                    encoder.classes_,
                    target=target,
                    split=split_name,
                )
                if use_excellent_blend:
                    adjusted_raw_m[target] = compute_metrics_from_label_strings(
                        y,
                        after,
                        encoder.classes_,
                        target=target,
                        split=split_name,
                    )
                model_m[target] = compute_metrics_from_label_strings(
                    y,
                    before,
                    encoder.classes_,
                    target=target,
                    split=split_name,
                )
                audit_m[target] = compute_rule_override_audit(
                    y,
                    before,
                    after,
                    encoder.classes_,
                    target=target,
                    split=split_name,
                )
                logger.info(
                    "Rule eval — split=%s target=%s | model macro-F1 %.4f → "
                    "adjusted %.4f (Δ %+.4f) | helpful=%d harmful=%d neutral=%d "
                    "override_precision=%s | excellent→model rows=%d",
                    split_name,
                    target,
                    model_m[target]["macro_f1"],
                    adjusted_m[target]["macro_f1"],
                    audit_m[target]["delta_macro_f1"],
                    audit_m[target]["n_helpful"],
                    audit_m[target]["n_harmful"],
                    audit_m[target]["n_neutral"],
                    audit_m[target]["override_precision"],
                    n_ex_subst,
                )
                grade_report_sections.append(
                    build_grade_analysis_report(
                        y,
                        before,
                        after,
                        encoder.classes_,
                        target=target,
                        split=split_name,
                        after_for_scoring=after_eval
                        if use_excellent_blend
                        else None,
                    )
                )

                # Rule-owned slice section (true-label-conditioned view)
                owned_for_target = rule_owned_grades.get(target, [])
                grade_report_sections.append(
                    build_rule_owned_slice_report(
                        y,
                        before,
                        after,
                        encoder.classes_,
                        target=target,
                        split=split_name,
                        rule_owned_grades=owned_for_target,
                    )
                )

                # Formatted override-audit section with by_after /
                # by_transition breakdowns (compact text tables).
                grade_report_sections.append(
                    format_override_audit_report(audit_m[target])
                )

                # Compute slice recalls for rule-owned grades — used
                # both for MLflow tags/metrics and for the baseline JSON.
                y_t2, combined_list, (y_b_idx, y_a_idx) = (
                    remap_true_and_encode_predictions(
                        y, encoder.classes_, before, after
                    )
                )
                combined_arr = __import__("numpy").array(combined_list)
                slice_recalls: dict[str, dict] = {}
                for g in owned_for_target:
                    slice_recalls[g] = {
                        "recall_model": slice_recall_for_grade(
                            y_t2, y_b_idx, combined_arr, g
                        ),
                        "recall_adjusted": slice_recall_for_grade(
                            y_t2, y_a_idx, combined_arr, g
                        ),
                    }
                audit_m[target]["slice_recall"] = slice_recalls
                baseline_snapshot.setdefault(split_name, {})[target] = {
                    "rule_owned_grades": owned_for_target,
                    "override_precision": audit_m[target][
                        "override_precision"
                    ],
                    "n_changed": audit_m[target]["n_changed"],
                    "n_helpful": audit_m[target]["n_helpful"],
                    "n_harmful": audit_m[target]["n_harmful"],
                    "n_neutral": audit_m[target]["n_neutral"],
                    "delta_macro_f1": audit_m[target].get("delta_macro_f1"),
                    "delta_accuracy": audit_m[target].get("delta_accuracy"),
                    "by_after": audit_m[target].get("by_after", {}),
                    "by_transition": audit_m[target].get(
                        "by_transition", {}
                    ),
                    "slice_recall": slice_recalls,
                }

            reports_dir = Path(self.config["paths"]["reports"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            gap = "\n\n" + "=" * 72 + "\n\n"
            ga_path = reports_dir / f"grade_analysis_{split_name}.txt"
            ga_path.write_text(
                gap.join(grade_report_sections),
                encoding="utf-8",
            )
            grade_analysis_paths[split_name] = str(ga_path)
            logger.info("Grade analysis report — %s", ga_path)

            out[split_name] = {
                "adjusted": adjusted_m,
                "model":    model_m,
                "audit":    audit_m,
                "coverage": coverage,
            }
            if use_excellent_blend:
                out[split_name]["adjusted_raw"] = adjusted_raw_m

            sk = split_name
            for target in ("sleeve", "media"):
                adj = adjusted_m[target]
                aud = audit_m[target]
                mlflow_flat[f"rule_adjusted_{sk}_{target}_macro_f1"] = float(
                    adj["macro_f1"]
                )
                mlflow_flat[f"rule_adjusted_{sk}_{target}_accuracy"] = float(
                    adj["accuracy"]
                )
                if use_excellent_blend:
                    rawm = adjusted_raw_m[target]
                    mlflow_flat[
                        f"rule_adjusted_raw_{sk}_{target}_macro_f1"
                    ] = float(rawm["macro_f1"])
                    mlflow_flat[
                        f"rule_adjusted_raw_{sk}_{target}_accuracy"
                    ] = float(rawm["accuracy"])
                mlflow_flat[f"rule_model_{sk}_{target}_macro_f1"] = float(
                    model_m[target]["macro_f1"]
                )
                mlflow_flat[f"rule_model_{sk}_{target}_accuracy"] = float(
                    model_m[target]["accuracy"]
                )
                mlflow_flat[f"rule_audit_{sk}_{target}_delta_macro_f1"] = float(
                    aud["delta_macro_f1"]
                )
                mlflow_flat[f"rule_audit_{sk}_{target}_delta_accuracy"] = float(
                    aud["delta_accuracy"]
                )
                mlflow_flat[f"rule_audit_{sk}_{target}_helpful"] = float(
                    aud["n_helpful"]
                )
                mlflow_flat[f"rule_audit_{sk}_{target}_harmful"] = float(
                    aud["n_harmful"]
                )
                mlflow_flat[f"rule_audit_{sk}_{target}_neutral"] = float(
                    aud["n_neutral"]
                )
                if aud["override_precision"] is not None:
                    mlflow_flat[
                        f"rule_audit_{sk}_{target}_override_precision"
                    ] = float(aud["override_precision"])

                # Rule-owned slice metrics (§6).
                for g, vals in (aud.get("slice_recall") or {}).items():
                    gsafe = g.lower().replace(" ", "_")
                    rm = vals.get("recall_model")
                    ra = vals.get("recall_adjusted")
                    if rm is not None:
                        mlflow_flat[
                            f"rule_slice_{sk}_{target}_true_{gsafe}_recall_model"
                        ] = float(rm)
                    if ra is not None:
                        mlflow_flat[
                            f"rule_slice_{sk}_{target}_true_{gsafe}_recall_adjusted"
                        ] = float(ra)

                # Stratified "harmful to <grade>" metrics, capped to
                # the top-K destinations that actually saw overrides.
                by_after = aud.get("by_after") or {}
                for g, row in by_after.items():
                    gsafe = g.lower().replace(" ", "_")
                    mlflow_flat[
                        f"rule_audit_{sk}_{target}_harmful_to_{gsafe}"
                    ] = float(row.get("n_harmful", 0))
                    prec = row.get("override_precision")
                    if prec is not None:
                        mlflow_flat[
                            f"rule_audit_{sk}_{target}_override_precision_to_{gsafe}"
                        ] = float(prec)

        if "test" in out:
            for k in ("sleeve", "media"):
                mlflow_flat[f"rule_adjusted_{k}_macro_f1"] = float(
                    out["test"]["adjusted"][k]["macro_f1"]
                )
                mlflow_flat[f"rule_adjusted_{k}_accuracy"] = float(
                    out["test"]["adjusted"][k]["accuracy"]
                )

        # --- §8 Baseline snapshot: JSON artifact + MLflow tags ----------
        if baseline_snapshot:
            reports_dir = Path(self.config["paths"]["reports"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            snap_path = reports_dir / "rule_engine_baseline.json"
            self._write_rule_engine_baseline(
                snap_path, baseline_snapshot
            )
            logger.info("Rule engine baseline snapshot — %s", snap_path)
            try:
                self._tag_rule_engine_baseline(baseline_snapshot)
            except Exception as exc:  # mlflow-optional path
                logger.debug(
                    "Skipped MLflow baseline tags (no active run?): %s", exc
                )

        return out, mlflow_flat, grade_analysis_paths

    # -----------------------------------------------------------------------
    # Baseline snapshot helpers (§8)
    # -----------------------------------------------------------------------
    @staticmethod
    def _current_git_sha() -> Optional[str]:
        """Return short git HEAD sha, or None if git is unavailable."""
        import subprocess

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return sha.decode("utf-8", errors="replace").strip() or None
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return None

    def _write_rule_engine_baseline(
        self, path: Path, snapshot: dict
    ) -> None:
        """Persist the rule-engine baseline snapshot as canonical JSON."""
        from datetime import datetime, timezone

        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "commit": self._current_git_sha(),
            "splits": snapshot,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _tag_rule_engine_baseline(snapshot: dict) -> None:
        """
        Mirror key baseline numbers to MLflow **tags** (stringified) so
        they are grep-able in the UI without polluting the metrics graph.
        Top-K cap from `by_after` naturally bounds the tag count.
        """
        if not mlflow.active_run():
            return
        for split_name, targets in snapshot.items():
            for target, data in targets.items():
                base = f"rule_baseline_{split_name}_{target}"
                prec = data.get("override_precision")
                if prec is not None:
                    mlflow.set_tag(f"{base}_override_precision", f"{prec:.4f}")
                mlflow.set_tag(
                    f"{base}_rule_owned_grades",
                    ",".join(data.get("rule_owned_grades", [])),
                )
                for g, row in (data.get("by_after") or {}).items():
                    gsafe = g.lower().replace(" ", "_")
                    mlflow.set_tag(
                        f"{base}_harmful_to_{gsafe}",
                        str(row.get("n_harmful", 0)),
                    )
                    rp = row.get("override_precision")
                    if rp is not None:
                        mlflow.set_tag(
                            f"{base}_override_precision_to_{gsafe}",
                            f"{rp:.4f}",
                        )
                for g, vals in (data.get("slice_recall") or {}).items():
                    gsafe = g.lower().replace(" ", "_")
                    ra = vals.get("recall_adjusted")
                    if ra is not None:
                        mlflow.set_tag(
                            f"{base}_true_{gsafe}_recall_adjusted",
                            f"{ra:.4f}",
                        )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Vinyl condition grader pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand ---
    train_parser = subparsers.add_parser(
        "train", help="Run the full training pipeline"
    )
    train_parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    train_parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    train_parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion — use existing raw data",
    )
    train_parser.add_argument(
        "--skip-ebay-ingest",
        action="store_true",
        help="Ingest Discogs only (omit eBay JP; needs DISCOGS_TOKEN only)",
    )
    train_parser.add_argument(
        "--skip-harmonize",
        action="store_true",
        help="Skip harmonization — use existing unified.jsonl",
    )
    train_parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing — use existing split files",
    )
    train_parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip TF-IDF feature extraction",
    )
    train_parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Train baseline only — skip transformer (alias for --skip-transformer)",
    )
    train_parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip DistilBERT training (step 6); use with promoted weights + rule eval on baseline if needed",
    )
    train_parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Skip TF-IDF baseline training (step 5); load baseline_*.pkl from "
            "paths.artifacts and evaluate on existing feature matrices"
        ),
    )
    train_parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow entirely (same as mlflow.enabled: false in config).",
    )
    train_parser.add_argument(
        "--mlflow-no-artifacts",
        action="store_true",
        help=(
            "Log params/metrics only — no artifact uploads or registry "
            "(mlflow.log_artifacts: false). Ignored with --no-mlflow."
        ),
    )
    train_parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip registering the transformer pyfunc to the MLflow model registry",
    )
    train_parser.add_argument(
        "--registry-model-name",
        default=None,
        help="Override mlflow.registry_model_name for this run",
    )
    train_parser.add_argument(
        "--skip-sale-history",
        action="store_true",
        help=(
            "After Discogs ingest, do not export sale_history SQLite to "
            "discogs_sale_history.jsonl (that export runs by default: feature-store enrich + "
            "vinyl filter)."
        ),
    )

    # --- predict subcommand ---
    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on text input"
    )
    predict_parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    predict_parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )

    # Mutually exclusive input — either --text or --file
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Raw text to grade (single prediction)",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to file with one text per line (batch prediction)",
    )

    predict_parser.add_argument(
        "--model",
        choices=["baseline", "transformer"],
        default=None,
        help="Model to use for inference (overrides config)",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions as JSONL (optional)",
    )

    args = parser.parse_args()

    # --- Execute ---
    if args.command == "train":
        pipeline = Pipeline(
            config_path=args.config,
            guidelines_path=args.guidelines,
        )
        train_kw = {}
        if args.no_register:
            train_kw["register_after_pipeline"] = False
        if args.registry_model_name:
            train_kw["registry_model_name_override"] = args.registry_model_name
        pipeline.train(
            skip_ingest=args.skip_ingest,
            skip_ebay_ingest=args.skip_ebay_ingest,
            skip_harmonize=args.skip_harmonize,
            skip_preprocess=args.skip_preprocess,
            skip_features=args.skip_features,
            skip_transformer=args.skip_transformer,
            baseline_only=args.baseline_only,
            skip_baseline=args.skip_baseline,
            no_mlflow=args.no_mlflow,
            mlflow_no_artifacts=args.mlflow_no_artifacts,
            skip_sale_history_ingest=args.skip_sale_history,
            **train_kw,
        )

    elif args.command == "predict":
        pipeline = Pipeline(
            config_path=args.config,
            guidelines_path=args.guidelines,
        )

        # Override inference model if specified
        if args.model:
            pipeline.infer_model = args.model

        if args.text:
            # Single prediction
            prediction = pipeline.predict(text=args.text)
            print(json.dumps(prediction, indent=2))

            if args.output:
                with open(args.output, "w") as f:
                    f.write(json.dumps(prediction) + "\n")

        elif args.file:
            # Batch prediction from file
            with open(args.file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

            predictions = pipeline.predict_batch(texts=texts)

            for pred in predictions:
                print(json.dumps(pred, indent=2))

            if args.output:
                with open(args.output, "w") as f:
                    for pred in predictions:
                        f.write(json.dumps(pred) + "\n")

                logger.info(
                    "Saved %d predictions to %s",
                    len(predictions),
                    args.output,
                )


if __name__ == "__main__":
    main()
