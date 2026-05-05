"""CLI for `python -m grader.src.models.transformer`."""

import logging

from grader.src.config_io import load_yaml_mapping

from .trainer_core import TransformerTrainer


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train two-head DistilBERT classifier "
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
        help="Train for 1 epoch without saving artifacts",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Unfreeze encoder for full fine-tuning (ablation)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Train and save artifacts without MLflow",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Same as --skip-mlflow / mlflow.enabled: false",
    )
    parser.add_argument(
        "--mlflow-no-artifacts",
        action="store_true",
        help=(
            "Log params/metrics only (mlflow.log_artifacts: false). "
            "Ignored with --skip-mlflow / --no-mlflow."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml_mapping(args.config)
    ml = cfg.setdefault("mlflow", {})
    if args.skip_mlflow or args.no_mlflow:
        ml["enabled"] = False
    elif args.mlflow_no_artifacts and ml.get("enabled", True):
        ml["log_artifacts"] = False

    trainer = TransformerTrainer(
        config_path=args.config,
        freeze_encoder_override=not args.unfreeze if args.unfreeze else None,
        config=cfg,
    )
    trainer.run(
        dry_run=args.dry_run,
        skip_mlflow=args.skip_mlflow or args.no_mlflow,
    )


if __name__ == "__main__":
    main()
