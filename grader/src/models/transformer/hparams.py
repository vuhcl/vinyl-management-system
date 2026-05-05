"""YAML merge helpers for transformer hyperparameter presets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .constants import _META_HPARAM_KEYS


def merge_transformer_hparams(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge transformer YAML subsection; nested media_evidence_aux combined."""
    out = dict(base)
    for k, v in overrides.items():
        if k in _META_HPARAM_KEYS:
            continue
        if k == "media_evidence_aux" and isinstance(v, dict):
            inner = dict(out.get("media_evidence_aux") or {})
            inner.update(v)
            out[k] = inner
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def resolve_model_artifact_dir(
    artifacts_dir: Path,
    artifact_subdir: Optional[str],
) -> Path:
    """
    Directory for transformer weights relative to ``paths.artifacts``.

    ``artifact_subdir`` should be a *relative* tail such as ``tuning/foo``.
    If someone passes the full ``grader/artifacts`` path again, it would
    double-resolve to ``.../grader/artifacts/grader/artifacts`` — strip
    the redundant prefix when it matches ``artifacts_dir``'s path parts.
    """
    ad = artifacts_dir
    if not artifact_subdir or not str(artifact_subdir).strip():
        return ad
    s = Path(artifact_subdir.strip())
    if s.is_absolute():
        return s.resolve()
    ar_parts = ad.parts
    sub_parts = s.parts
    if len(sub_parts) >= len(ar_parts) and sub_parts[: len(ar_parts)] == ar_parts:
        tail = sub_parts[len(ar_parts) :]
        return ad if not tail else ad / Path(*tail)
    return ad / s
