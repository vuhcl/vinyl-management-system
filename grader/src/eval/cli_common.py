"""
Shared argparse helpers for grader ``eval`` CLIs.

Depends only on :mod:`grader.src.config_io` (no heavy grader imports).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from grader.src.config_io import load_yaml_mapping

DEFAULT_GRADER_CONFIG = "grader/configs/grader.yaml"
DEFAULT_GRADING_GUIDELINES = "grader/configs/grading_guidelines.yaml"


def add_grader_config_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_GRADER_CONFIG,
    help_text: str = "Path to grader config YAML",
) -> None:
    parser.add_argument("--config", default=default, help=help_text)


def add_grading_guidelines_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_GRADING_GUIDELINES,
    help_text: str = "Path to grading guidelines YAML",
) -> None:
    parser.add_argument("--guidelines", default=default, help=help_text)


def load_grader_config_mapping(config_path: str | Path) -> dict[str, Any]:
    """Load ``grader.yaml`` as a mapping (raises if not a YAML object)."""
    return load_yaml_mapping(config_path)
