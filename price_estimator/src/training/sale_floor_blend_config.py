"""``SaleFloorBlendConfig`` + YAML merge + listing/y label caps."""
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

from .sale_floor_enums import SaleConditionPolicy
from .sale_floor_row_parsing import _resolve_grade_delta_path

@dataclass(frozen=True)
class SaleFloorBlendConfig:
    n_min_trend: int = 8
    recency_half_life_days: float = 365.0
    w_base: float = 0.55
    w_min: float = 0.2
    w_max: float = 0.9
    tier_b_delta: float = 0.05
    tier_c_delta: float = 0.1
    gap_epsilon_log: float = 0.02
    gap_k_down: float = 0.15
    gap_k_up: float = 0.12
    gap_delta_cap: float = 0.5
    nm_substrings: tuple[str, ...] = (
        "near mint",
        "(nm",
        "mint (m",
    )
    # --- ordinal cascade + uplift (legacy default: substring NM only, no uplift)
    sale_condition_policy: str = SaleConditionPolicy.NM_SUBSTRINGS_ONLY.value
    strict_min_ordinal: float = 7.0
    relax_steps: tuple[float, ...] = (6.0, 5.0)
    min_rows_strict: int = 8
    min_rows_relax_1: int = 5
    min_rows_relax_2: int = 6
    apply_grade_uplift_to_nm: bool = False
    uplift_nm_media_ordinal: float = 7.0
    uplift_nm_sleeve_ordinal: float = 7.0
    base_alpha: float = 0.06
    base_beta: float = 0.04
    ref_grade: float = 8.0
    grade_delta_scale: GradeDeltaScaleParams | None = None
    # --- label sanity caps (bad listing floors / extrapolated nowcasts)
    label_cap_enabled: bool = True
    label_cap_sale_nowcast_max_multiple_of_max_price: float = 15.0
    label_cap_listing_max_multiple_of_sale_peak: float = 15.0
    # Pre-cap raw listing floor before ``_cap_listing_floor_against_sales`` (disabled if ≤ 0).
    label_cap_listing_lo_clip_multiple_of_sale_peak: float = 2.0
    label_cap_y_max_multiple_of_sale_peak: float = 15.0
    label_cap_y_max_multiple_of_anchor_m: float = 30.0
    label_cap_listing_max_multiple_of_median_only: float = 40.0
    # --- optional sold-nowcast stabilization (omit `nowcast_stability` / defaults = legacy math)
    trend_time_bin: str = "day"
    trend_bin_agg: str = "median"
    min_history_span_days: float = 0.0
    tier_a_shrink_lambda: float = 1.0
    tier_b_center: str = "mean"


def sale_floor_blend_config_from_raw(
    raw_cfg: dict[str, Any],
    *,
    nm_grade_key: str,
) -> SaleFloorBlendConfig:
    merged: dict[str, Any] = dict(raw_cfg)
    oc = raw_cfg.get("ordinal_cascade")
    if isinstance(oc, dict):
        merged = {**merged, **oc}

    nm_tup = merged.get("nm_substrings")
    nm_sub: tuple[str, ...] = SaleFloorBlendConfig.nm_substrings
    if isinstance(nm_tup, (list, tuple)) and nm_tup:
        nm_sub = tuple(str(x) for x in nm_tup)

    policy_raw = str(
        merged.get("sale_condition_policy", SaleConditionPolicy.NM_SUBSTRINGS_ONLY),
    ).strip().lower()
    try:
        policy = SaleConditionPolicy(policy_raw).value
    except ValueError:
        policy = SaleConditionPolicy.NM_SUBSTRINGS_ONLY.value

    rs = merged.get("relax_steps")
    if isinstance(rs, (list, tuple)) and rs:
        relax_steps = tuple(float(x) for x in rs)
    else:
        relax_steps = (6.0, 5.0)

    ca = merged.get("condition_adjustment")
    if isinstance(ca, dict):
        base_alpha = float(ca.get("alpha", 0.06))
        base_beta = float(ca.get("beta", 0.04))
        ref_grade = float(ca.get("ref_grade", 8.0))
    else:
        base_alpha = float(merged.get("base_alpha", 0.06))
        base_beta = float(merged.get("base_beta", 0.04))
        ref_grade = float(merged.get("ref_grade", 8.0))

    scale_map: dict[str, Any] = {}
    gds = merged.get("grade_delta_scale")
    if isinstance(gds, dict):
        scale_map.update(gds)
    gpath = _resolve_grade_delta_path(
        str(merged["grade_delta_scale_path"]).strip()
        if merged.get("grade_delta_scale_path")
        else None
    )
    if gpath is not None and gpath.is_file():
        try:
            blob = json.loads(gpath.read_text())
            if isinstance(blob, dict):
                inner = blob.get("grade_delta_scale")
                if isinstance(inner, dict):
                    scale_map.update(inner)
                elif any(
                    k in blob
                    for k in (
                        "price_gamma",
                        "price_ref_usd",
                        "age_k",
                        "age_center_year",
                    )
                ):
                    scale_map.update(blob)
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    gparams = GradeDeltaScaleParams.from_mapping(scale_map if scale_map else None)

    lc = merged.get("label_cap")
    if isinstance(lc, dict):
        label_cap_enabled = bool(lc.get("enabled", True))
        sn_max = float(
            lc.get("sale_nowcast_max_multiple_of_max_price", 30.0),
        )
        lm_sale = float(lc.get("listing_max_multiple_of_sale_peak", 35.0))
        lo_clip = float(
            lc.get(
                "listing_lo_clip_multiple_of_sale_peak",
                SaleFloorBlendConfig.label_cap_listing_lo_clip_multiple_of_sale_peak,
            ),
        )
        ym_sp = float(lc.get("y_max_multiple_of_sale_peak", 20.0))
        ym_m = float(lc.get("y_max_multiple_of_anchor_m", 50.0))
        lm_med = float(lc.get("listing_max_multiple_of_median_only", 40.0))
    else:
        label_cap_enabled = bool(merged.get("label_cap_enabled", True))
        sn_max = float(
            merged.get("label_cap_sale_nowcast_max_multiple_of_max_price", 30.0),
        )
        lm_sale = float(
            merged.get("label_cap_listing_max_multiple_of_sale_peak", 35.0),
        )
        lo_clip = float(
            merged.get(
                "label_cap_listing_lo_clip_multiple_of_sale_peak",
                SaleFloorBlendConfig.label_cap_listing_lo_clip_multiple_of_sale_peak,
            ),
        )
        ym_sp = float(merged.get("label_cap_y_max_multiple_of_sale_peak", 20.0))
        ym_m = float(merged.get("label_cap_y_max_multiple_of_anchor_m", 50.0))
        lm_med = float(
            merged.get("label_cap_listing_max_multiple_of_median_only", 40.0),
        )

    um = float(merged.get("uplift_nm_media_ordinal", 7.0))
    us = float(merged.get("uplift_nm_sleeve_ordinal", 7.0))
    if bool(merged.get("uplift_nm_from_price_suggestion_grade")):
        o = condition_string_to_ordinal(nm_grade_key)
        if o >= 0:
            um = us = float(o)

    ns = merged.get("nowcast_stability")
    trend_time_bin = SaleFloorBlendConfig.trend_time_bin
    trend_bin_agg = SaleFloorBlendConfig.trend_bin_agg
    min_history_span_days = SaleFloorBlendConfig.min_history_span_days
    tier_a_shrink_lambda = SaleFloorBlendConfig.tier_a_shrink_lambda
    tier_b_center = SaleFloorBlendConfig.tier_b_center
    if isinstance(ns, dict):
        ttb = str(ns.get("trend_time_bin", trend_time_bin)).strip().lower()
        if ttb in ("day", "week", "month"):
            trend_time_bin = ttb
        tba = str(ns.get("trend_bin_agg", trend_bin_agg)).strip().lower()
        if tba in ("median", "mean", "last"):
            trend_bin_agg = tba
        min_history_span_days = float(ns.get("min_history_span_days", min_history_span_days))
        tier_a_shrink_lambda = float(ns.get("tier_a_shrink_lambda", tier_a_shrink_lambda))
        tbc = str(ns.get("tier_b_center", tier_b_center)).strip().lower()
        if tbc in ("mean", "weighted_median"):
            tier_b_center = tbc

    return SaleFloorBlendConfig(
        n_min_trend=int(merged.get("n_min_trend", 8)),
        recency_half_life_days=float(merged.get("recency_half_life_days", 365.0)),
        w_base=float(merged.get("w_base", 0.55)),
        w_min=float(merged.get("w_min", 0.2)),
        w_max=float(merged.get("w_max", 0.9)),
        tier_b_delta=float(merged.get("tier_b_delta", 0.05)),
        tier_c_delta=float(merged.get("tier_c_delta", 0.1)),
        gap_epsilon_log=float(merged.get("gap_epsilon_log", 0.02)),
        gap_k_down=float(merged.get("gap_k_down", 0.15)),
        gap_k_up=float(merged.get("gap_k_up", 0.12)),
        gap_delta_cap=float(merged.get("gap_delta_cap", 0.5)),
        nm_substrings=nm_sub,
        sale_condition_policy=policy,
        strict_min_ordinal=float(merged.get("strict_min_ordinal", 7.0)),
        relax_steps=relax_steps,
        min_rows_strict=int(merged.get("min_rows_strict", 8)),
        min_rows_relax_1=int(merged.get("min_rows_relax_1", 5)),
        min_rows_relax_2=int(merged.get("min_rows_relax_2", 6)),
        apply_grade_uplift_to_nm=bool(merged.get("apply_grade_uplift_to_nm", False)),
        uplift_nm_media_ordinal=um,
        uplift_nm_sleeve_ordinal=us,
        base_alpha=base_alpha,
        base_beta=base_beta,
        ref_grade=ref_grade,
        grade_delta_scale=gparams,
        label_cap_enabled=label_cap_enabled,
        label_cap_sale_nowcast_max_multiple_of_max_price=sn_max,
        label_cap_listing_max_multiple_of_sale_peak=lm_sale,
        label_cap_listing_lo_clip_multiple_of_sale_peak=lo_clip,
        label_cap_y_max_multiple_of_sale_peak=ym_sp,
        label_cap_y_max_multiple_of_anchor_m=ym_m,
        label_cap_listing_max_multiple_of_median_only=lm_med,
        trend_time_bin=trend_time_bin,
        trend_bin_agg=trend_bin_agg,
        min_history_span_days=min_history_span_days,
        tier_a_shrink_lambda=tier_a_shrink_lambda,
        tier_b_center=tier_b_center,
    )


def _cap_listing_floor_against_sales(
    lo: float,
    s: float | None,
    p_max_obs: float | None,
    *,
    cfg: SaleFloorBlendConfig,
) -> float:
    """Pull absurd listing floors toward observed sold comps before log-blend."""
    mult = float(cfg.label_cap_listing_max_multiple_of_sale_peak)
    base = 0.0
    if s is not None and math.isfinite(s) and s > 0:
        base = max(float(s), float(p_max_obs or 0.0))
    elif p_max_obs is not None and p_max_obs > 0:
        base = float(p_max_obs)
    else:
        return lo
    cap = base * mult
    return float(min(lo, cap)) if cap > 0 and math.isfinite(cap) else lo


def _cap_final_y_label(
    y: float,
    m_anchor: float,
    p_max_obs: float | None,
    *,
    cfg: SaleFloorBlendConfig,
) -> float:
    """Upper-cap blended label vs sale peak and vs residual anchor ``m``."""
    out = float(y)
    if p_max_obs is not None and p_max_obs > 0:
        hi_sp = float(p_max_obs) * float(cfg.label_cap_y_max_multiple_of_sale_peak)
        if math.isfinite(hi_sp) and hi_sp > 0:
            out = min(out, hi_sp)
    if m_anchor is not None and math.isfinite(m_anchor) and m_anchor > 0:
        hi_m = float(m_anchor) * float(cfg.label_cap_y_max_multiple_of_anchor_m)
        if math.isfinite(hi_m) and hi_m > 0:
            out = min(out, hi_m)
    return max(out, 1e-9)
