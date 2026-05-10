"""Merge optional keys into ``model_manifest.json`` without clobbering existing layout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .regressor_constants import MANIFEST_FILE


def merge_model_manifest(model_dir: Path | str, updates: dict[str, Any]) -> None:
    """Read manifest JSON if present, shallow-merge ``updates``, write back."""
    d = Path(model_dir)
    path = d / MANIFEST_FILE
    raw: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except (json.JSONDecodeError, OSError):
            raw = {}
    raw.update(updates)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
