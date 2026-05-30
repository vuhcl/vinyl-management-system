"""Tests for grader YAML config loading (inherits merge)."""
from __future__ import annotations

from pathlib import Path

from grader.src.config_io import load_yaml_mapping


def test_grader_yaml_inherits_base_and_keeps_relative_paths() -> None:
    cfg = load_yaml_mapping("grader/configs/grader.yaml")
    assert "inherits" not in cfg
    assert cfg["project"]["name"] == "vinyl_collector_ai"
    assert cfg["data"]["splits"]["random_seed"] == 42
    assert cfg["paths"]["raw"] == "grader/data/raw/"


def test_grader_yaml_resolves_from_absolute_path() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_yaml_mapping(root / "grader/configs/grader.yaml")
    assert cfg["project"]["name"] == "vinyl_collector_ai"
