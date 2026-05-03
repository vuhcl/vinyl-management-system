"""Persisted condition adjustment parameters (log-space shifts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_params() -> dict[str, Any]:
    return {
        "alpha": 0.06,
        "beta": 0.04,
        "ref_grade": 8.0,
        "grade_delta_scale": None,
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


def merge_grade_delta_scale_dict(base: dict[str, Any] | None, overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``grade_delta_scale`` mappings (YAML / fitted JSON overlay)."""
    out = dict(base) if base else {}
    cur = out.get("grade_delta_scale")
    cur_d: dict[str, Any] = dict(cur) if isinstance(cur, dict) else {}
    cur_d.update(overlay)
    out["grade_delta_scale"] = cur_d
    return out


def merge_inference_condition_params(
    artifact: dict[str, Any],
    yaml_overlay: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Merge YAML ``ordinal_cascade`` condition scalers over artefact-backed params.

    Precedence: **YAML wins** where keys are present so shipped ``configs/base.yaml``
    stays aligned with ``fit_grade_delta_scale`` / training edits without rewriting
    ``condition_params.json`` in the champion bundle on every coef tweak.
    """
    if not yaml_overlay:
        return dict(artifact)
    out = dict(artifact)
    for k in ("alpha", "beta", "ref_grade"):
        if k not in yaml_overlay:
            continue
        vals = yaml_overlay[k]
        if vals is None:
            continue
        try:
            out[k] = float(vals)
        except (TypeError, ValueError):
            continue
    gds = yaml_overlay.get("grade_delta_scale")
    if isinstance(gds, dict) and gds:
        out = merge_grade_delta_scale_dict(out, dict(gds))
    return out


def _grade_delta_overlay_from_fit_file(blob: dict[str, Any]) -> dict[str, Any]:
    """Strip ``schema_version`` / ``fit_metadata`` so only scaler keys merge into ``grade_delta_scale``."""
    inner = blob.get("grade_delta_scale")
    if isinstance(inner, dict):
        return dict(inner)
    keys = (
        "price_ref_usd",
        "price_gamma",
        "price_scale_min",
        "price_scale_max",
        "age_k",
        "age_center_year",
    )
    return {k: blob[k] for k in keys if k in blob}


def load_params_with_grade_delta_overlays(model_dir: Path | str) -> dict[str, Any]:
    """
    Load ``condition_params.json`` plus optional ``grade_delta_scale.json`` in the same
    directory (emit-only fit artifact), merged into ``grade_delta_scale``.
    """
    d = Path(model_dir)
    out = load_params(d / "condition_params.json")
    gf = d / "grade_delta_scale.json"
    if not gf.is_file():
        return out
    try:
        blob = json.loads(gf.read_text())
    except (json.JSONDecodeError, OSError, TypeError):
        return out
    if not isinstance(blob, dict) or not blob:
        return out
    overlay = _grade_delta_overlay_from_fit_file(blob)
    if not overlay:
        return out
    merged = merge_grade_delta_scale_dict(out, overlay)
    for coef in ("alpha", "beta"):
        if coef not in blob or blob[coef] is None:
            continue
        try:
            merged[coef] = float(blob[coef])
        except (TypeError, ValueError):
            pass
    return merged
