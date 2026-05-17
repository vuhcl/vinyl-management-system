"""Load monitoring threshold YAML and validate required keys."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_NUMERIC = ("year", "decade", "label_tier", "pred_log1p_usd")
_REQUIRED_CAT = ("genre", "country")
_OPTIONAL_ROOT = (
    "version",
    "numeric",
    "categorical",
    "binary",
    "integrality",
    "challenge_detect",
)


def default_thresholds_path() -> Path:
    return Path(__file__).resolve().parent / "thresholds.yaml"


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    p = path or default_thresholds_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("thresholds YAML must be a mapping at root")
    for k in _REQUIRED_NUMERIC:
        if "numeric" not in data or k not in data.get("numeric", {}):
            raise KeyError(f"Missing numeric.{k} in {p}")
    for k in _REQUIRED_CAT:
        if "categorical" not in data or k not in data.get("categorical", {}):
            raise KeyError(f"Missing categorical.{k} in {p}")
    if "pred_log1p_usd" not in data.get("numeric", {}):
        raise KeyError("Missing numeric.pred_log1p_usd")
    return data
