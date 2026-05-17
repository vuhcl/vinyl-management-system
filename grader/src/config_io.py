"""
YAML loading helpers for the grader package.

Uses only the standard library plus PyYAML — no imports from other grader
modules (avoids cycles with pipeline, models, mlflow helpers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: Path | str, *, encoding: str = "utf-8") -> Any:
    """Load YAML from ``path``. May return ``None`` if the document is empty."""
    p = Path(path)
    with p.open(encoding=encoding) as f:
        return yaml.safe_load(f)


def load_yaml_mapping(path: Path | str, *, encoding: str = "utf-8") -> dict[str, Any]:
    """
    Load YAML and require a top-level mapping (dict).

    Raises:
        TypeError: if the document is not a JSON-like object / mapping.
    """
    doc = load_yaml(path, encoding=encoding)
    if not isinstance(doc, Mapping):
        raise TypeError(
            f"Expected YAML mapping at {path!s}, got {type(doc).__name__}"
        )
    return dict(doc)
