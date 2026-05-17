"""CLI entrypoint for `python -m grader.src.models.baseline`."""

import logging

from grader.src.config_io import load_yaml_mapping

from .model import BaselineModel


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train and evaluate two-head LR baseline "
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
        help="Train and evaluate without saving artifacts",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow (mlflow.enabled: false)",
    )
    parser.add_argument(
        "--mlflow-no-artifacts",
        action="store_true",
        help=(
            "Params/metrics only (mlflow.log_artifacts: false). "
            "Ignored with --no-mlflow."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml_mapping(args.config)
    ml = cfg.setdefault("mlflow", {})
    if args.no_mlflow:
        ml["enabled"] = False
    elif args.mlflow_no_artifacts and ml.get("enabled", True):
        ml["log_artifacts"] = False

    model = BaselineModel(config_path=args.config, config=cfg)
    model.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
