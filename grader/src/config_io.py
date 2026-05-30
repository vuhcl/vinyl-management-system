"""
YAML loading helpers for the grader package.

Uses only the standard library plus PyYAML — no imports from other grader
modules (avoids cycles with pipeline, models, mlflow helpers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def _repo_root() -> Path:
    """Monorepo root (``grader/src/config_io.py`` → two levels up)."""
    return Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _load_yaml_with_inherits(path: Path, *, root: Path) -> dict[str, Any]:
    """
    Load YAML; if ``inherits`` is set (path relative to repo root), deep-merge
    parent first so the current file overrides. ``inherits`` is removed from result.
    """
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
        base = _load_yaml_with_inherits(parent_path, root=root)
        _deep_merge(base, cfg)
        return base
    return cfg


def load_yaml(path: Path | str, *, encoding: str = "utf-8") -> Any:
    """Load YAML from ``path``. May return ``None`` if the document is empty."""
    p = Path(path)
    with p.open(encoding=encoding) as f:
        return yaml.safe_load(f)


def load_yaml_mapping(path: Path | str, *, encoding: str = "utf-8") -> dict[str, Any]:
    """
    Load YAML and require a top-level mapping (dict).

    Honors ``inherits`` chains relative to the monorepo root without absolutizing
    grader ``paths.*`` values.

    Raises:
        TypeError: if the document is not a JSON-like object / mapping.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _repo_root() / p
    doc = _load_yaml_with_inherits(p, root=_repo_root())
    if not isinstance(doc, Mapping):
        raise TypeError(
            f"Expected YAML mapping at {path!s}, got {type(doc).__name__}"
        )
    return dict(doc)
