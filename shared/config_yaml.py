"""Shared YAML ``inherits`` merge (no path absolutization)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge_yaml(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge_yaml(base[key], value)
        else:
            base[key] = value


def load_yaml_mapping_with_inherits(
    path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {}
    parent = cfg.pop("inherits", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = root / parent_path
        base = load_yaml_mapping_with_inherits(parent_path, root=root)
        deep_merge_yaml(base, cfg)
        return base
    return cfg


def require_mapping(doc: object, *, path: Path | str) -> dict[str, Any]:
    if not isinstance(doc, Mapping):
        raise TypeError(
            f"Expected YAML mapping at {path!s}, got {type(doc).__name__}"
        )
    return dict(doc)
