"""Load grader demo golden JSON files for scripts and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def golden_predict_demo_path() -> Path:
    return repo_root() / "grader" / "demo" / "golden_predict_demo.json"


def load_golden_predict_demo(path: Path | None = None) -> dict[str, Any]:
    p = path or golden_predict_demo_path()
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object JSON in {p}")
    return data
