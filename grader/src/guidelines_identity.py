"""
Stable identity helpers for ``grading_guidelines.yaml``.

Used by RuleEngine, Preprocessor, baselines, MLflow tags, and serving checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

# Monotonic rubric revision label (YYYY.MM.DD or YYYY.MM.DD.n same-day bumps).
_GUIDELINES_VERSION_RE = re.compile(
    r"^\d{4}\.\d{2}\.\d{2}(\.\d+)?$"
)


def guidelines_version_from_mapping(g: Mapping[str, Any]) -> str:
    """Return ``guidelines_version`` string or ``unknown`` if missing/blank."""
    v = g.get("guidelines_version")
    if v is None:
        return "unknown"
    s = str(v).strip()
    return s if s else "unknown"


def canonical_grades_sha256_from_mapping(g: Mapping[str, Any]) -> str:
    """
    Fingerprint canonical sleeve/media grade lists for Tier C drift detection.
    """
    sg = list(g.get("sleeve_grades") or [])
    mg = list(g.get("media_grades") or [])
    payload = json.dumps(
        {"media_grades": mg, "sleeve_grades": sg},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_valid_guidelines_version_format(version: str) -> bool:
    """Whether ``version`` matches the repo convention (not ``unknown``)."""
    if version == "unknown":
        return False
    return bool(_GUIDELINES_VERSION_RE.match(version))


def guidelines_path_basename(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    name = p.name
    return name if name else None
