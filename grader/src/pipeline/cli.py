"""``python -m grader.src.pipeline`` argument parsing and execution."""

from __future__ import annotations

import argparse
import json
import logging

from .model import Pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Vinyl condition grader pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        help=(
            "Train baseline only — skip transformer (alias for "
            "--skip-transformer)"
        ),
    )
    train_parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help=(
            "Skip DistilBERT training (step 6); use with promoted weights + "
            "rule eval on baseline if needed"
        ),
    )
    train_parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Skip TF-IDF baseline training (step 5); load baseline_*.pkl "
            "from paths.artifacts and evaluate on existing feature matrices"
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
        help=(
            "Skip registering the transformer pyfunc to the MLflow model "
            "registry"
        ),
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
            "discogs_sale_history.jsonl (that export runs by default: "
            "feature-store enrich + vinyl filter)."
        ),
    )

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

    if args.command == "train":
        pipeline = Pipeline(
            config_path=args.config,
            guidelines_path=args.guidelines,
        )
        train_kw: dict = {}
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

        if args.model:
            pipeline.infer_model = args.model

        if args.text:
            prediction = pipeline.predict(text=args.text)
            print(json.dumps(prediction, indent=2))

            if args.output:
                with open(args.output, "w") as f:
                    f.write(json.dumps(prediction) + "\n")

        elif args.file:
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
