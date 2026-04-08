"""Persisted condition adjustment parameters (log-space shifts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_params() -> dict[str, Any]:
    return {
        "alpha": -0.06,
        "beta": -0.04,
        "ref_grade": 8.0,
    }


def save_params(path: Path | str, params: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params, indent=2))


def load_params(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return default_params()
    return {**default_params(), **json.loads(p.read_text())}
