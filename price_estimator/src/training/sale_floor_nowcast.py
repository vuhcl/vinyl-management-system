"""Sold-dollar nowcast ``s`` (tier A/B/C)."""
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

