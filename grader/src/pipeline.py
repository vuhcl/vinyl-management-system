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

from grader.src.data.harmonize_labels import LabelHarmonizer
from grader.src.data.ingest_discogs import DiscogsIngester
from grader.src.data.ingest_ebay import EbayIngester
from grader.src.data.preprocess import Preprocessor
from grader.src.evaluation.calibration import CalibrationEvaluator
from grader.src.evaluation.metrics import (
    compare_models,
    compare_models_per_class,
    log_comparison_to_mlflow,
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

        # MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

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
            self._rule_engine = RuleEngine(
                guidelines_path=self.guidelines_path
            )
        return self._rule_engine

    def _get_tfidf(self) -> TFIDFFeatureBuilder:
        if self._tfidf is None:
            self._tfidf = TFIDFFeatureBuilder(
                config_path=self.config_path
            )
        return self._tfidf

    # -----------------------------------------------------------------------
    # Training pipeline
    # -----------------------------------------------------------------------
    def train(
        self,
        skip_ingest: bool = False,
        skip_harmonize: bool = False,
        skip_preprocess: bool = False,
        skip_features: bool = False,
        skip_transformer: bool = False,
        baseline_only: bool = False,
    ) -> dict:
        """
        Run the full training pipeline end to end.

        Steps:
          1. Ingest Discogs and eBay JP data
          2. Harmonize labels into unified dataset
          3. Preprocess text and split train/val/test
          4. Build TF-IDF features
          5. Train and evaluate baseline model
          6. Train and evaluate transformer model (skippable)
          7. Compare baseline vs transformer metrics
          8. Generate calibration plots for both models
          9. Compute rule engine coverage on test split

        Args:
            skip_ingest:      skip steps 1 — use existing raw data
            skip_harmonize:   skip step 2 — use existing unified.jsonl
            skip_preprocess:  skip step 3 — use existing split files
            skip_features:    skip step 4 — use existing feature matrices
            skip_transformer: skip step 6 — baseline only
            baseline_only:    alias for skip_transformer=True

        Returns:
            Dict with results from all completed steps.
        """
        skip_transformer = skip_transformer or baseline_only
        results = {}

        with mlflow.start_run(run_name="full_training_pipeline"):

            # Step 1 — Ingestion
            if not skip_ingest:
                logger.info("=" * 50)
                logger.info("STEP 1 — DATA INGESTION")
                logger.info("=" * 50)

                discogs_ingester = DiscogsIngester(
                    config_path=self.config_path,
                    guidelines_path=self.guidelines_path,
                )
                discogs_ingester.run()

                ebay_ingester = EbayIngester(
                    config_path=self.config_path,
                    guidelines_path=self.guidelines_path,
                )
                ebay_ingester.run()
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
                )
                results["preprocess"] = preprocessor.run()
            else:
                logger.info("Skipping preprocessing — using existing splits.")

            # Step 4 — TF-IDF features
            if not skip_features:
                logger.info("=" * 50)
                logger.info("STEP 4 — TF-IDF FEATURE EXTRACTION")
                logger.info("=" * 50)

                tfidf = TFIDFFeatureBuilder(config_path=self.config_path)
                results["features"] = tfidf.run()
            else:
                logger.info("Skipping feature extraction — using existing matrices.")

            # Step 5 — Baseline model
            logger.info("=" * 50)
            logger.info("STEP 5 — BASELINE MODEL (TF-IDF + LR)")
            logger.info("=" * 50)

            baseline = BaselineModel(config_path=self.config_path)
            baseline_results = baseline.run()
            results["baseline"] = baseline_results

            baseline_test_metrics = {
                target: baseline_results["eval"]["test"][target]
                for target in ["sleeve", "media"]
            }

            # Step 6 — Transformer model
            transformer_test_metrics = None

            if not skip_transformer:
                logger.info("=" * 50)
                logger.info("STEP 6 — TRANSFORMER MODEL (DistilBERT)")
                logger.info("=" * 50)

                trainer = TransformerTrainer(config_path=self.config_path)
                transformer_results = trainer.run()
                results["transformer"] = transformer_results

                transformer_test_metrics = {
                    target: transformer_results["eval"]["test"][target]
                    for target in ["sleeve", "media"]
                }
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

                print("\n" + comparison_table)
                print(per_class_table)

                # Save comparison tables as artifacts
                comparison_path = (
                    self.artifacts_dir / "model_comparison.txt"
                )
                with open(comparison_path, "w") as f:
                    f.write(comparison_table + "\n\n" + per_class_table)

                mlflow.log_artifact(str(comparison_path))
                log_comparison_to_mlflow(
                    baseline_metrics=baseline_test_metrics,
                    transformer_metrics=transformer_test_metrics,
                )

                results["comparison"] = {
                    "table":     comparison_table,
                    "per_class": per_class_table,
                }

            # Step 8 — Calibration plots
            logger.info("=" * 50)
            logger.info("STEP 8 — CALIBRATION EVALUATION")
            logger.info("=" * 50)

            calibration_evaluator = CalibrationEvaluator(
                config_path=self.config_path
            )

            # Load test features for calibration plots
            tfidf_builder = self._get_tfidf()
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
                    log_to_mlflow=True,
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
                        log_to_mlflow=True,
                    )

            # Step 9 — Rule engine coverage on test split
            logger.info("=" * 50)
            logger.info("STEP 9 — RULE ENGINE COVERAGE")
            logger.info("=" * 50)

            rule_engine  = self._get_rule_engine()
            test_records = self._load_split("test")
            test_texts   = [
                r.get("text_clean") or r.get("text", "")
                for r in test_records
            ]

            # Use transformer predictions if available, else baseline
            if transformer_test_metrics is not None:
                test_predictions = trainer.predict(
                    texts=test_texts,
                    item_ids=[r.get("item_id") for r in test_records],
                    records=test_records,
                )
            else:
                # Rebuild baseline predictions on test set
                test_predictions = self._baseline_predict_from_features(
                    baseline, target="sleeve"
                )

            coverage = rule_engine.coverage_report(
                test_predictions, test_texts
            )

            logger.info(
                "Rule engine coverage — overrides: %d (%.1f%%) | "
                "contradictions: %d (%.1f%%)",
                coverage["overrides_applied"],
                coverage["override_rate"] * 100,
                coverage["contradictions"],
                coverage["contradiction_rate"] * 100,
            )

            mlflow.log_metrics(
                {
                    "rule_override_rate":      coverage["override_rate"],
                    "rule_contradiction_rate": coverage["contradiction_rate"],
                    "rule_overrides_applied":  coverage["overrides_applied"],
                    "rule_contradictions":     coverage["contradictions"],
                }
            )

            results["rule_coverage"] = coverage

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
            text_clean       = preprocessor.clean_text(text)
            clean_texts.append(text_clean)

            record = {
                "item_id":         item_ids[i],
                "text":            text,
                "text_clean":      text_clean,
                "media_verifiable": media_verifiable,
                "source":          meta.get("source", "user_input"),
                **meta,
            }
            records.append(record)

        # Step 2 — Model prediction
        predictions = self._model_predict(
            clean_texts=clean_texts,
            item_ids=item_ids,
            records=records,
        )

        # Step 3 — Rule engine post-processing
        final_predictions = rule_engine.apply_batch(
            predictions=predictions,
            texts=clean_texts,
        )

        return final_predictions

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

        # Vectorize texts using fitted vectorizers
        X_sleeve = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_sleeve.pkl")
        ).transform(clean_texts)

        X_media = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_media.pkl")
        ).transform(clean_texts)

        return self._baseline.predict(
            X_sleeve=X_sleeve,
            X_media=X_media,
            item_ids=item_ids,
            records=records,
        )

    def _baseline_predict_from_features(
        self,
        baseline: BaselineModel,
        target: str,
    ) -> list[dict]:
        """
        Helper to rebuild baseline predictions from saved test features.
        Used in rule engine coverage step when transformer is not run.
        """
        X_test, _ = TFIDFFeatureBuilder.load_features(
            str(self.artifacts_dir / "features"),
            split="test",
            target=target,
        )
        test_records = self._load_split("test")
        item_ids = [r.get("item_id", str(i)) for i, r in enumerate(test_records)]

        X_sleeve = X_test if target == "sleeve" else None
        X_media  = X_test if target == "media"  else None

        # Load both if not provided
        if X_sleeve is None:
            X_sleeve, _ = TFIDFFeatureBuilder.load_features(
                str(self.artifacts_dir / "features"),
                split="test",
                target="sleeve",
            )
        if X_media is None:
            X_media, _ = TFIDFFeatureBuilder.load_features(
                str(self.artifacts_dir / "features"),
                split="test",
                target="media",
            )

        return baseline.predict(
            X_sleeve=X_sleeve,
            X_media=X_media,
            item_ids=item_ids,
            records=test_records,
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
        help="Train baseline only — skip transformer",
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
        pipeline.train(
            skip_ingest=args.skip_ingest,
            skip_harmonize=args.skip_harmonize,
            skip_preprocess=args.skip_preprocess,
            skip_features=args.skip_features,
            baseline_only=args.baseline_only,
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
