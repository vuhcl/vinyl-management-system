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

from grader.src.config_io import load_yaml_mapping
from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    vinyl_grader_pyfunc_has_python_model,
)
from grader.src.data.preprocess import Preprocessor
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer
from grader.src.pipeline_train_steps import (
    apply_train_mlflow_env_overrides,
    run_steps_1_through_4,
    run_train_steps_5_through_9,
)
from grader.src.rules.rule_engine import RuleEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
        self.config          = load_yaml_mapping(config_path)

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
        if True:
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
                preprocessor.compute_description_quality(
                    text,
                    text_clean,
                    sleeve_label=str(meta.get("sleeve_label") or ""),
                    media_label=str(meta.get("media_label") or ""),
                )
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
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
