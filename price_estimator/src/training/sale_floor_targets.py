"""§7.1d sold nowcast ``s`` + listing floor blend for ``training_label.mode: sale_floor_blend``."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import theilslopes

from price_estimator.src.features.vinyliq_features import (
    GradeDeltaScaleParams,
    condition_string_to_ordinal,
    grade_delta_scale_params_from_cond,
    log1p_nm_equivalent_from_sale_usd,
)
from price_estimator.src.storage.marketplace_db import price_suggestion_values_by_grade

_PRICE_ESTIMATOR_ROOT = Path(__file__).resolve().parents[2]


def _parse_ps_grade(raw_json: str | None, grade_key: str) -> float | None:
    from price_estimator.src.training.label_synthesis import (
        parse_price_suggestion_value,
    )

    return parse_price_suggestion_value(raw_json, grade_key)


def _positive(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


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
    sale_condition_policy: str = "nm_substrings_only"
    strict_min_ordinal: float = 7.0
    relax_steps: tuple[float, ...] = (6.0, 5.0)
    min_rows_strict: int = 8
    min_rows_relax_1: int = 5
    min_rows_relax_2: int = 6
    apply_grade_uplift_to_nm: bool = False
    uplift_nm_media_ordinal: float = 7.0
    uplift_nm_sleeve_ordinal: float = 7.0
    base_alpha: float = -0.06
    base_beta: float = -0.04
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

    policy = (
        str(merged.get("sale_condition_policy", "nm_substrings_only")).strip().lower()
    )
    if policy not in ("nm_substrings_only", "ordinal_cascade"):
        policy = "nm_substrings_only"

    rs = merged.get("relax_steps")
    if isinstance(rs, (list, tuple)) and rs:
        relax_steps = tuple(float(x) for x in rs)
    else:
        relax_steps = (6.0, 5.0)

    ca = merged.get("condition_adjustment")
    if isinstance(ca, dict):
        base_alpha = float(ca.get("alpha", -0.06))
        base_beta = float(ca.get("beta", -0.04))
        ref_grade = float(ca.get("ref_grade", 8.0))
    else:
        base_alpha = float(merged.get("base_alpha", -0.06))
        base_beta = float(merged.get("base_beta", -0.04))
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


def _iso_week_bucket_center(dt: datetime) -> datetime:
    y, w, _ = dt.isocalendar()
    return datetime.fromisocalendar(y, w, 4).replace(
        hour=12, minute=0, second=0, microsecond=0
    )


def _month_bucket_center(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 15, 12, 0, 0)


def _tier_a_bucket_key(dt: datetime, *, trend_time_bin: str) -> tuple[Any, ...]:
    if trend_time_bin == "month":
        return (dt.year, dt.month)
    return dt.isocalendar()[:2]


def _tier_a_bucket_center_for_date(dt: datetime, *, trend_time_bin: str) -> datetime:
    if trend_time_bin == "month":
        return _month_bucket_center(dt)
    return _iso_week_bucket_center(dt)


def _aggregate_bin_prices(values: list[float], dates_in_bin: list[datetime], agg: str) -> float:
    arr = np.array(values, dtype=np.float64)
    if agg == "mean":
        return float(np.mean(arr))
    if agg == "last":
        latest = max(range(len(dates_in_bin)), key=lambda i: dates_in_bin[i])
        return float(values[latest])
    return float(np.median(arr))


def _weighted_median(prices: np.ndarray, weights: np.ndarray) -> float | None:
    if prices.size == 0:
        return None
    sw = float(np.sum(weights))
    if sw <= 1e-18:
        med = float(np.median(prices))
        return med if med > 0 else None
    order = np.argsort(prices)
    p_sorted = prices[order]
    w_sorted = weights[order]
    cw = np.cumsum(w_sorted)
    half = 0.5 * sw
    idx = int(np.searchsorted(cw, half, side="left"))
    idx = min(idx, int(p_sorted.size - 1))
    out = float(p_sorted[idx])
    return out if out > 0 else None


def _tier_b_center(prices: np.ndarray, weights: np.ndarray, mode: str) -> float | None:
    mode_l = mode.strip().lower()
    if mode_l == "weighted_median":
        return _weighted_median(prices, weights)
    sw = float(np.sum(weights))
    if sw <= 1e-18:
        med = float(np.median(prices))
        return med if med > 0 else None
    return float(np.sum(prices * weights) / sw)


def _tier_a_xy_for_theil(
    eligible: list[tuple[datetime, float]],
    t_ref: datetime,
    *,
    t0: datetime,
    cfg: SaleFloorBlendConfig,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Build ``(xs, ys=log price)`` and evaluation abscissa ``t_x`` for ``t_ref``."""
    bin_kind = cfg.trend_time_bin.strip().lower()
    if bin_kind == "day":
        dates = [d for d, _ in eligible]
        prices = np.array([p for _, p in eligible], dtype=np.float64)
        xs = np.array([(d - t0).total_seconds() / 86400.0 for d in dates], dtype=np.float64)
        ys = np.log(prices)
        t_x = (t_ref - t0).total_seconds() / 86400.0
        return xs, ys, float(t_x)

    buckets: dict[tuple[Any, ...], list[tuple[datetime, float]]] = {}
    for d, p in eligible:
        if p <= 0 or not math.isfinite(float(p)):
            continue
        key = _tier_a_bucket_key(d, trend_time_bin=bin_kind)
        buckets.setdefault(key, []).append((d, float(p)))

    if not buckets:
        return None

    centers: list[datetime] = []
    y_vals: list[float] = []
    for key in sorted(buckets.keys()):
        rows = buckets[key]
        ds = [x[0] for x in rows]
        pv = [x[1] for x in rows]
        c0 = rows[0][0]
        center = _tier_a_bucket_center_for_date(c0, trend_time_bin=bin_kind)
        centers.append(center)
        y_vals.append(_aggregate_bin_prices(pv, ds, cfg.trend_bin_agg.strip().lower()))

    xs = np.array([(c - t0).total_seconds() / 86400.0 for c in centers], dtype=np.float64)
    y_arr = np.array(y_vals, dtype=np.float64)
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr <= 0):
        return None
    ys = np.log(y_arr)
    cref = _tier_a_bucket_center_for_date(t_ref, trend_time_bin=bin_kind)
    t_x = (cref - t0).total_seconds() / 86400.0
    return xs, ys, float(t_x)


def sold_nowcast_s(
    eligible: list[tuple[datetime, float]],
    t_ref: datetime,
    *,
    cfg: SaleFloorBlendConfig,
) -> tuple[float | None, str, int]:
    """
    Tier A/B/C sold-dollar nowcast ``s`` at ``t_ref``.

    Returns ``(s_or_none, tier_name, n_eligible)``.
    """
    n = len(eligible)
    if n == 0:
        return None, "none", 0
    prices = np.array([p for _, p in eligible], dtype=np.float64)
    dates = [d for d, _ in eligible]
    t0 = min(dates)

    span_days = (max(dates) - min(dates)).total_seconds() / 86400.0
    min_span = float(cfg.min_history_span_days)
    tier_a_allowed = min_span <= 0.0 or span_days >= min_span

    series = _tier_a_xy_for_theil(eligible, t_ref, t0=t0, cfg=cfg)
    xs: np.ndarray | None = None
    ys: np.ndarray | None = None
    t_x = 0.0
    if series is not None:
        xs, ys, t_x = series

    n_trend = int(xs.shape[0]) if xs is not None else 0
    if (
        tier_a_allowed
        and xs is not None
        and ys is not None
        and n_trend >= cfg.n_min_trend
        and float(np.std(xs)) > 1e-9
    ):
        try:
            res = theilslopes(ys, xs)
            intercept = float(res.intercept)
            slope = float(res.slope)
            log_s = intercept + slope * float(t_x)
            s = float(math.exp(log_s))
            if s > 0 and math.isfinite(s):
                lam = float(cfg.tier_a_shrink_lambda)
                lam = max(0.0, min(1.0, lam))
                if lam < 1.0:
                    ages_days = np.array(
                        [(t_ref - d).total_seconds() / 86400.0 for d in dates],
                        dtype=np.float64,
                    )
                    H = max(1.0, float(cfg.recency_half_life_days))
                    w = np.exp(-np.maximum(ages_days, 0.0) / H)
                    anchor = _weighted_median(prices, w)
                    if anchor is not None and anchor > 0 and math.isfinite(anchor):
                        log_s = lam * math.log(s) + (1.0 - lam) * math.log(float(anchor))
                        s = float(math.exp(log_s))
                if s > 0 and math.isfinite(s):
                    return s, "A", n
        except (ValueError, RuntimeError):
            pass

    if n >= 3:
        ages_days = np.array(
            [(t_ref - d).total_seconds() / 86400.0 for d in dates], dtype=np.float64
        )
        H = max(1.0, float(cfg.recency_half_life_days))
        w = np.exp(-np.maximum(ages_days, 0.0) / H)
        sw = float(np.sum(w))
        if sw <= 0:
            med = float(np.median(prices))
            return med if med > 0 else None, "B", n
        yb = _tier_b_center(prices, w, cfg.tier_b_center)
        return yb if yb is not None and yb > 0 else None, "B", n

    last_p = float(prices[-1])
    return last_p if last_p > 0 else None, "C", n


def effective_listing_floor_lo(row: dict[str, Any]) -> float | None:
    return _positive(row.get("release_lowest_price"))


def max_price_suggestion_ladder_usd(row: dict[str, Any]) -> float | None:
    """§7.1d residual anchor: max positive grade value from ``price_suggestions_json``."""
    vals = price_suggestion_values_by_grade(row.get("price_suggestions_json"))
    if not vals:
        return None
    return max(vals.values())


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


def blend_weight_w_eff(
    *,
    s: float,
    lo: float,
    tier: str,
    cfg: SaleFloorBlendConfig,
) -> float:
    w = float(cfg.w_base)
    if tier == "B":
        w -= float(cfg.tier_b_delta)
    elif tier == "C":
        w -= float(cfg.tier_c_delta)
    w = max(float(cfg.w_min), min(float(cfg.w_max), w))

    eps = float(cfg.gap_epsilon_log)
    d_log = math.log(lo) - math.log(s)
    if d_log < -eps:
        w += float(cfg.gap_k_down) * min(abs(d_log), float(cfg.gap_delta_cap))
    elif d_log > eps:
        w -= float(cfg.gap_k_up) * min(d_log, float(cfg.gap_delta_cap))

    return max(float(cfg.w_min), min(float(cfg.w_max), w))


def sale_floor_blend_y(
    s: float | None,
    lo: float | None,
    tier: str,
    *,
    cfg: SaleFloorBlendConfig,
) -> float | None:
    """§7.1d log-blend; no ``lo`` → ``y = s``; no ``s`` → None (caller excludes or uses other mode)."""
    if s is not None and s > 0 and lo is not None and lo > 0:
        w_eff = blend_weight_w_eff(s=s, lo=lo, tier=tier, cfg=cfg)
        return float(math.exp(w_eff * math.log(s) + (1.0 - w_eff) * math.log(lo)))
    if s is not None and s > 0:
        return float(s)
    if lo is not None and lo > 0:
        return float(lo)
    return None


def _sale_floor_blend_compute(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    sf_cfg: dict[str, Any],
    nm_grade_key: str,
    release_year: float | None = None,
) -> tuple[float | None, float | None, dict[str, float], dict[str, Any]]:
    """
    Core sale-floor blend; returns ``(y_label, m_anchor, x_flags, diagnostics)``.

    ``diagnostics`` is for QA (listing vs sold nowcast vs caps); safe to log, not used in training X.
    """
    raw_cfg = sf_cfg if isinstance(sf_cfg, dict) else {}
    cfg = sale_floor_blend_config_from_raw(raw_cfg, nm_grade_key=nm_grade_key)

    lo = effective_listing_floor_lo(mp_row)
    lo_raw_usd = float(lo) if lo is not None and lo > 0 else None
    has_listing_floor = 1.0 if lo is not None and lo > 0 else 0.0

    sh_ok = (
        fetch_status is not None
        and str(fetch_status.get("status") or "").strip().lower() == "ok"
    )
    sh_fetched = str(fetch_status.get("fetched_at") or "") if fetch_status else None
    t_ref = reference_time_t_ref(str(mp_row.get("fetched_at") or ""), sh_fetched)

    s: float | None = None
    tier = "none"
    n_elig = 0
    s_imputed = 0.0
    has_sale_history = 0.0
    relax_tag = "n/a"
    elig: list[tuple[datetime, float]] = []

    if t_ref is not None and sh_ok:
        if cfg.sale_condition_policy == "ordinal_cascade":
            elig, relax_tag = eligible_ordinal_cascade_sale_rows(
                sale_rows,
                t_ref,
                cfg=cfg,
                mp_row=mp_row,
                nm_grade_key=nm_grade_key,
                release_year=release_year,
            )
        else:
            elig = eligible_nm_sale_rows(sale_rows, t_ref, cfg=cfg)
            relax_tag = "legacy_nm_substrings"

        s, tier, n_elig = sold_nowcast_s(elig, t_ref, cfg=cfg)
        if s is not None and s > 0:
            has_sale_history = 1.0

    p_max_obs: float | None = None
    if elig:
        _prices = np.array([float(p) for _, p in elig], dtype=np.float64)
        if _prices.size > 0:
            p_max_obs = float(np.max(_prices))

    s_pre_cap = float(s) if s is not None and s > 0 else None
    if (
        cfg.label_cap_enabled
        and s is not None
        and s > 0
        and p_max_obs is not None
        and p_max_obs > 0
    ):
        hi_now = p_max_obs * float(cfg.label_cap_sale_nowcast_max_multiple_of_max_price)
        if math.isfinite(hi_now) and hi_now > 0:
            s = min(float(s), hi_now)

    lo_for_blend: float | None = lo
    listing_lo_clip_applied = False
    if cfg.label_cap_enabled and lo_for_blend is not None and lo_for_blend > 0:
        k_clip = float(cfg.label_cap_listing_lo_clip_multiple_of_sale_peak)
        if (
            p_max_obs is not None
            and p_max_obs > 0
            and k_clip > 0
            and math.isfinite(k_clip)
            and math.isfinite(float(lo_for_blend))
        ):
            hi_clip = float(p_max_obs) * k_clip
            if hi_clip > 0 and math.isfinite(hi_clip) and float(lo_for_blend) > hi_clip:
                lo_for_blend = hi_clip
                listing_lo_clip_applied = True
        lo_for_blend = _cap_listing_floor_against_sales(
            float(lo_for_blend),
            s,
            p_max_obs,
            cfg=cfg,
        )
        if p_max_obs is None and (s is None or s <= 0):
            med_only = _positive(mp_row.get("release_lowest_price"))
            if med_only is not None and med_only > 0:
                lo_for_blend = min(
                    lo_for_blend,
                    float(med_only)
                    * float(cfg.label_cap_listing_max_multiple_of_median_only),
                )

    y_blend = sale_floor_blend_y(s, lo_for_blend, tier, cfg=cfg)
    diag: dict[str, Any] = {
        "listing_floor_raw_usd": lo_raw_usd,
        "listing_floor_for_blend_usd": float(lo_for_blend)
        if lo_for_blend is not None and lo_for_blend > 0
        else None,
        "sold_nowcast_usd": float(s) if s is not None and s > 0 else None,
        "sold_nowcast_usd_pre_cap": s_pre_cap,
        "sold_tier": tier,
        "n_eligible_sales": int(n_elig),
        "p_max_sale_observed_usd": p_max_obs,
        "sale_history_fetch_ok": bool(sh_ok),
        "sale_relax_tag": relax_tag,
        "release_lowest_price_mp_usd": _positive(mp_row.get("release_lowest_price")),
        "y_blend_usd": float(y_blend) if y_blend is not None and y_blend > 0 else None,
        "listing_lo_clip_applied": listing_lo_clip_applied,
    }

    flags = {
        "has_sale_history": has_sale_history,
        "s_imputed": s_imputed,
        "has_listing_floor": has_listing_floor,
        "sale_relax_tier_code": _relax_tier_code(relax_tag),
    }

    if y_blend is None:
        return None, None, flags, diag

    y_f = float(y_blend)
    if has_sale_history:
        m = residual_anchor_m_full_data(mp_row, nm_grade_key=nm_grade_key)
    else:
        m = residual_anchor_m_no_sale_history(mp_row, nm_grade_key=nm_grade_key)
    if m is None:
        m = y_f
    m_f = float(m)
    if cfg.label_cap_enabled:
        y_final = _cap_final_y_label(y_f, m_f, p_max_obs, cfg=cfg)
    else:
        y_final = y_f
    diag["m_anchor_usd"] = float(m_f)
    diag["y_label_final_usd"] = float(y_final)
    diag["label_cap_enabled"] = bool(cfg.label_cap_enabled)
    return (
        float(y_final),
        float(m_f),
        flags,
        diag,
    )


def sale_floor_label_diagnostics(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    sf_cfg: dict[str, Any],
    nm_grade_key: str,
    release_year: float | None = None,
) -> tuple[float | None, float | None, dict[str, float], dict[str, Any]]:
    """
    Training-time sale-floor bundle plus a **diagnostic** dict for label QA.

    The diagnostic keys include ``listing_floor_raw_usd``, ``sold_nowcast_usd``,
    ``p_max_sale_observed_usd``, ``listing_lo_clip_applied``, ``y_label_final_usd``,
    ``m_anchor_usd``, etc.
    """
    return _sale_floor_blend_compute(
        mp_row,
        sale_rows,
        fetch_status,
        sf_cfg=sf_cfg,
        nm_grade_key=nm_grade_key,
        release_year=release_year,
    )


def sale_floor_blend_bundle(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    sf_cfg: dict[str, Any],
    nm_grade_key: str,
    release_year: float | None = None,
) -> tuple[float | None, float | None, dict[str, float]]:
    """
    Returns ``(y_label, m_anchor, x_flags)``.

    ``x_flags`` includes ``has_sale_history``, ``s_imputed``, ``has_listing_floor`` (0/1 floats).
    """
    y, m, flags, _diag = _sale_floor_blend_compute(
        mp_row,
        sale_rows,
        fetch_status,
        sf_cfg=sf_cfg,
        nm_grade_key=nm_grade_key,
        release_year=release_year,
    )
    return y, m, flags


def sale_floor_blend_sf_cfg_for_policy(
    sf_cfg: dict[str, Any] | None,
    policy: str,
) -> dict[str, Any]:
    """Shallow copy of ``sale_floor_blend`` YAML dict with ``sale_condition_policy`` set."""
    raw = dict(sf_cfg) if isinstance(sf_cfg, dict) else {}
    pol = str(policy).strip().lower()
    if pol not in ("nm_substrings_only", "ordinal_cascade"):
        pol = "nm_substrings_only"
    out = {**raw, "sale_condition_policy": pol}
    return out


def _relax_tier_code(tag: str) -> float:
    return {
        "strict": 0.0,
        "relax_1": 1.0,
        "relax_2": 2.0,
        "none": 3.0,
        "legacy_nm_substrings": -1.0,
        "n/a": -2.0,
    }.get(tag, -3.0)
