"""
YAML loading helpers for the grader package.

Uses only the standard library plus PyYAML — no imports from other grader
modules (avoids cycles with pipeline, models, mlflow helpers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from shared.config_yaml import load_yaml_mapping_with_inherits, require_mapping


def _repo_root() -> Path:
    """Monorepo root (``grader/src/config_io.py`` → two levels up)."""
    return Path(__file__).resolve().parents[2]


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
    doc = load_yaml_mapping_with_inherits(p, root=_repo_root())
    return require_mapping(doc, path=path)
