"""Serving-time anchors and PS ladder helpers."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from scipy.stats import theilslopes

from price_estimator.src.features.vinyliq_features import (
    GradeDeltaScaleParams,
    condition_string_to_ordinal,
    grade_delta_scale_params_from_cond,
    log1p_nm_equivalent_from_sale_usd,
)
from price_estimator.src.storage.marketplace_db import (
    price_suggestion_usd_for_grade_label,
    price_suggestion_values_by_grade,
)

from .sale_floor_row_parsing import _parse_ps_grade, _positive

def effective_listing_floor_lo(row: dict[str, Any]) -> float | None:
    return _positive(row.get("release_lowest_price"))


def max_price_suggestion_ladder_usd(row: dict[str, Any]) -> float | None:
    """§7.1d residual anchor: max positive grade value from ``price_suggestions_json``."""
    vals = price_suggestion_values_by_grade(row.get("price_suggestions_json"))
    if not vals:
        return None
    return max(vals.values())


def pre_uplift_grade_anchor_usd(row: dict[str, Any], *, nm_grade_key: str) -> float:
    """
    USD anchor for grade-delta scaling (not sold-nowcast ``s``).

    Prefers listing low, PS ladder, NM suggestion. (Legacy Discogs
    ``median_price`` was a mirror of ``release_lowest_price`` and has been
    retired; the listing-floor branch produces the same value.)
    """
    lo = effective_listing_floor_lo(row)
    if lo is not None:
        return float(lo)
    mx = max_price_suggestion_ladder_usd(row)
    if mx is not None:
        return float(mx)
    y_nm = _parse_ps_grade(row.get("price_suggestions_json"), nm_grade_key)
    if y_nm is not None:
        return float(y_nm)
    return 1.0


def residual_anchor_m_full_data(
    row: dict[str, Any],
    *,
    nm_grade_key: str,
) -> float | None:
    """
    ``m`` when sale history exists (§7.1d): max PS ladder → NM grade → listing floor.
    """
    mx = max_price_suggestion_ladder_usd(row)
    if mx is not None:
        return mx
    y_nm = _parse_ps_grade(row.get("price_suggestions_json"), nm_grade_key)
    if y_nm is not None:
        return float(y_nm)
    lo = effective_listing_floor_lo(row)
    if lo is not None:
        return float(lo)
    return None


def residual_anchor_m_no_sale_history(
    row: dict[str, Any], *, nm_grade_key: str
) -> float | None:
    """§7.1b-style: ``lo``-first; then NM suggestion; avoid PS max ladder when no SH."""
    lo = effective_listing_floor_lo(row)
    if lo is not None:
        return float(lo)
    y_nm = _parse_ps_grade(row.get("price_suggestions_json"), nm_grade_key)
    if y_nm is not None:
        return float(y_nm)
    return None


def _normalized_price_suggestions_json(mp_row: dict[str, Any]) -> str | None:
    psj = mp_row.get("price_suggestions_json")
    if isinstance(psj, dict):
        return json.dumps(psj, ensure_ascii=False, separators=(",", ":"))
    if psj is not None:
        x = str(psj).strip()
        return x or None
    return None


def inference_price_suggestion_ladder(mp_row: dict[str, Any]) -> dict[str, float]:
    """Parsed Discogs ladder (grade → positive USD); empty when missing/unparseable."""
    return price_suggestion_values_by_grade(
        _normalized_price_suggestions_json(mp_row),
    )


def inference_price_suggestion_anchor_usd_for_side(
    mp_row: dict[str, Any],
    *,
    role: Literal["media", "sleeve"],
    media_condition: str | None,
    sleeve_condition: str | None,
) -> float | None:
    """
    Single-side ladder USD for **media-only** vs **sleeve-only** price-suggestion roles.

    ``role=="media"`` uses ``media_condition`` on the ladder; ``role=="sleeve"`` uses
    ``sleeve_condition``.
    """
    ladder = inference_price_suggestion_ladder(mp_row)
    if not ladder:
        return None
    label = media_condition if role == "media" else sleeve_condition
    x = price_suggestion_usd_for_grade_label(ladder, label)
    if x is not None and x > 0 and math.isfinite(float(x)):
        return float(x)
    return None


def inference_residual_anchor_usd(
    mp_row: dict[str, Any],
    *,
    nm_grade_key: str,
) -> float | None:
    """
    Serving-time ``m`` for ``residual_log_median``: mirror training cascade when PS
    ladder has any positive grades (``has_sale_history`` branch); otherwise mirror
    the no-sale-history cascade (listing floor prioritised).
    """
    if max_price_suggestion_ladder_usd(mp_row) is not None:
        return residual_anchor_m_full_data(mp_row, nm_grade_key=nm_grade_key)
    return residual_anchor_m_no_sale_history(mp_row, nm_grade_key=nm_grade_key)


def inference_price_suggestion_condition_anchor_usd(
    mp_row: dict[str, Any],
    *,
    media_condition: str | None,
    sleeve_condition: str | None,
) -> float | None:
    """
    Per-request anchor: **minimum** of media and sleeve ladder USD (legacy helper).

    Serving uses :func:`inference_price_suggestion_anchor_usd_for_side` twice
    (media role + sleeve role) and averages reconstructed prices instead.
    """
    ladder = inference_price_suggestion_ladder(mp_row)
    if not ladder:
        return None
    m = price_suggestion_usd_for_grade_label(ladder, media_condition)
    s = price_suggestion_usd_for_grade_label(ladder, sleeve_condition)
    vals: list[float] = []
    for v in (m, s):
        if v is not None and v > 0 and math.isfinite(float(v)):
            vals.append(float(v))
    if not vals:
        return None
    return float(min(vals))
