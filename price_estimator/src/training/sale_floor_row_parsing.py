"""Row parsing, time reference, and PS-grade helpers for sale-floor logic."""
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

from .numeric_coercion import strictly_positive_float as _positive

_PRICE_ESTIMATOR_ROOT = Path(__file__).resolve().parents[2]

def _parse_ps_grade(raw_json: str | None, grade_key: str) -> float | None:
    from price_estimator.src.training.label_synthesis import (
        parse_price_suggestion_value,
    )

    return parse_price_suggestion_value(raw_json, grade_key)


def parse_iso_datetime(s: str | None) -> datetime | None:
    if s is None or not str(s).strip():
        return None
    t = str(s).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def reference_time_t_ref(
    mp_fetched_at: str | None,
    sh_fetch_fetched_at: str | None,
) -> datetime | None:
    """§7.1d: ``min(MP.fetched_at, SH_fetch.fetched_at)`` when both exist; else whichever is set."""
    mp = parse_iso_datetime(mp_fetched_at)
    sh = parse_iso_datetime(sh_fetch_fetched_at)
    if mp is not None and sh is not None:
        return min(mp, sh)
    return mp or sh


def sale_row_usd(row: dict[str, Any]) -> float | None:
    v = row.get("price_user_usd_approx")
    if v is not None:
        p = _positive(v)
        if p is not None:
            return p
    for key in ("price_user_currency_text", "price_original_text"):
        raw = row.get(key)
        if raw is None or not str(raw).strip():
            continue
        m = re.search(r"[\d,]+\.?\d*", str(raw).replace(",", ""))
        if not m:
            continue
        try:
            x = float(m.group(0))
        except ValueError:
            continue
        if x > 0:
            return x
    return None


def _nm_allowed(
    media: str | None,
    sleeve: str | None,
    *,
    nm_substrings: tuple[str, ...],
) -> bool:
    blob = f"{media or ''} {sleeve or ''}".lower()
    return any(s.lower() in blob for s in nm_substrings)


def effective_sale_condition_ordinal(media: str | None, sleeve: str | None) -> float:
    """Conservative sale grade: ``min(media_ord, sleeve_ord)`` (same ladder as training)."""
    ma = condition_string_to_ordinal(media)
    sl = condition_string_to_ordinal(sleeve)
    return min(ma, sl)


def _resolve_grade_delta_path(raw: str | None) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    p = Path(str(raw).strip())
    if p.is_file():
        return p
    if not p.is_absolute():
        cand = _PRICE_ESTIMATOR_ROOT / p
        if cand.is_file():
            return cand
    return p if p.is_file() else None

