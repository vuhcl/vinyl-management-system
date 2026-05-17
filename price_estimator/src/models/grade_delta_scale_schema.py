"""Schema helpers for ``grade_delta_scale.json`` (offline fit artifact)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .grade_delta_artifact_keys import GRADE_DELTA_FIT_TOP_LEVEL_NUMERIC_KEYS


def build_placeholder_grade_delta_fit(
    *,
    price_ref_usd: float = 50.0,
    price_gamma: float = 0.0,
    age_k: float = 0.0,
    note: str = "placeholder",
) -> dict[str, Any]:
    """Default / bootstrap artifact (no pooled sale fit)."""
    return {
        "schema_version": 1,
        "fit_metadata": {
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "row_count_sales": 0,
        },
        "price_ref_usd": float(price_ref_usd),
        "price_gamma": float(price_gamma),
        "price_scale_min": 0.25,
        "price_scale_max": 4.0,
        "age_k": float(age_k),
        "age_center_year": 2000.0,
    }


def write_grade_delta_scale_json(
    path: Path | str,
    blob: dict[str, Any],
) -> None:
    validate_grade_delta_scale_fit_json(blob)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blob, indent=2))


def validate_grade_delta_scale_fit_json(blob: Any) -> None:
    """Raise ``ValueError`` if ``blob`` is not a v1 fit / placeholder artifact."""
    if not isinstance(blob, dict) or not blob:
        raise ValueError(
            "grade_delta_scale fit JSON must be a non-empty object",
        )
    if int(blob.get("schema_version", -1)) != 1:
        raise ValueError("schema_version must be 1")
    meta = blob.get("fit_metadata")
    if not isinstance(meta, dict):
        raise ValueError("fit_metadata must be an object")
    if "fitted_at" not in meta:
        raise ValueError("fit_metadata.fitted_at is required")
    for key in GRADE_DELTA_FIT_TOP_LEVEL_NUMERIC_KEYS:
        if key not in blob:
            raise ValueError(f"missing top-level scaler key: {key}")
        try:
            float(blob[key])
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key} must be numeric") from e
    for opt in ("alpha", "beta"):
        if opt not in blob or blob[opt] is None:
            continue
        try:
            float(blob[opt])
        except (TypeError, ValueError) as e:
            raise ValueError(f"{opt} must be numeric when present") from e
