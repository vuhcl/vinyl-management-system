"""CLI entry for ``python -m grader.src.data.preprocess``."""

from __future__ import annotations

import argparse
import logging

from .preprocessor_core import Preprocessor


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Preprocess unified vinyl grader dataset"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and split without writing output files",
    )
    args = parser.parse_args()

    preprocessor = Preprocessor(
        config_path=args.config,
        guidelines_path=args.guidelines,
    )
    preprocessor.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

