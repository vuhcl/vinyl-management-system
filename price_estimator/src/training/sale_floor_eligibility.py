"""Eligible NM / ordinal-cascade sale rows for nowcast pools."""
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

from .sale_floor_blend_config import SaleFloorBlendConfig
from .sale_floor_inference import pre_uplift_grade_anchor_usd
from .sale_floor_row_parsing import (
    _nm_allowed,
    parse_iso_datetime,
    sale_row_usd,
)

def eligible_nm_sale_rows(
    rows: Iterable[dict[str, Any]],
    t_ref: datetime,
    *,
    cfg: SaleFloorBlendConfig,
) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    for r in rows:
        if not _nm_allowed(
            r.get("media_condition"),
            r.get("sleeve_condition"),
            nm_substrings=cfg.nm_substrings,
        ):
            continue
        price = sale_row_usd(r)
        if price is None:
            continue
        od = parse_iso_datetime(str(r.get("order_date") or ""))
        if od is None or od > t_ref:
            continue
        out.append((od, float(price)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _sale_row_candidates(
    rows: Iterable[dict[str, Any]],
    t_ref: datetime,
    *,
    min_effective_ord: float,
) -> list[tuple[datetime, float, float, float]]:
    """``(order_date, usd, media_ord, sleeve_ord)`` with ``min(media,sleeve) >= min_effective_ord``."""
    out: list[tuple[datetime, float, float, float]] = []
    for r in rows:
        mo = condition_string_to_ordinal(r.get("media_condition"))
        so = condition_string_to_ordinal(r.get("sleeve_condition"))
        eff = min(mo, so)
        if eff < min_effective_ord:
            continue
        price = sale_row_usd(r)
        if price is None:
            continue
        od = parse_iso_datetime(str(r.get("order_date") or ""))
        if od is None or od > t_ref:
            continue
        out.append((od, float(price), float(mo), float(so)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _usd_after_optional_uplift(
    usd: float,
    media_ord: float,
    sleeve_ord: float,
    *,
    cfg: SaleFloorBlendConfig,
    anchor_usd: float,
    release_year: float | None,
) -> float:
    if not cfg.apply_grade_uplift_to_nm:
        return float(usd)
    log_nm = log1p_nm_equivalent_from_sale_usd(
        usd,
        media_ord,
        sleeve_ord,
        cfg.uplift_nm_media_ordinal,
        cfg.uplift_nm_sleeve_ordinal,
        base_alpha=cfg.base_alpha,
        base_beta=cfg.base_beta,
        anchor_usd=anchor_usd,
        release_year=release_year,
        scale_params=cfg.grade_delta_scale,
    )
    adj = float(math.expm1(min(max(log_nm, 0.0), 25.0)))
    return adj if adj > 0 else float(usd)


def eligible_ordinal_cascade_sale_rows(
    rows: Iterable[dict[str, Any]],
    t_ref: datetime,
    *,
    cfg: SaleFloorBlendConfig,
    mp_row: dict[str, Any],
    nm_grade_key: str,
    release_year: float | None,
) -> tuple[list[tuple[datetime, float]], str]:
    """
    Ordinal cascade pools (strict → VG+ → VG) on ``min(media_ord, sleeve_ord)``.

    Returns ``(eligible_for_nowcast, relax_tag)`` where ``relax_tag`` is
    ``strict`` | ``relax_1`` | ``relax_2`` | ``none``.
    """
    anchor = pre_uplift_grade_anchor_usd(mp_row, nm_grade_key=nm_grade_key)

    strict = _sale_row_candidates(
        rows, t_ref, min_effective_ord=float(cfg.strict_min_ordinal)
    )
    if len(strict) >= int(cfg.min_rows_strict):
        elig = [
            (
                d,
                _usd_after_optional_uplift(
                    p, m, s, cfg=cfg, anchor_usd=anchor, release_year=release_year
                ),
            )
            for d, p, m, s in strict
        ]
        return elig, "strict"

    floors = list(cfg.relax_steps)
    floor1 = float(floors[0]) if floors else 6.0
    pool1 = _sale_row_candidates(rows, t_ref, min_effective_ord=floor1)
    if len(pool1) >= int(cfg.min_rows_relax_1):
        elig = [
            (
                d,
                _usd_after_optional_uplift(
                    p, m, s, cfg=cfg, anchor_usd=anchor, release_year=release_year
                ),
            )
            for d, p, m, s in pool1
        ]
        return elig, "relax_1"

    floor2 = float(floors[1]) if len(floors) > 1 else 5.0
    pool2 = _sale_row_candidates(rows, t_ref, min_effective_ord=floor2)
    if len(pool2) >= int(cfg.min_rows_relax_2):
        elig = [
            (
                d,
                _usd_after_optional_uplift(
                    p, m, s, cfg=cfg, anchor_usd=anchor, release_year=release_year
                ),
            )
            for d, p, m, s in pool2
        ]
        return elig, "relax_2"

    return [], "none"

