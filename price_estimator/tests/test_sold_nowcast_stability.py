"""Sold nowcast ``sold_nowcast_s`` optional stabilization (binning, span, shrink, Tier B center)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from price_estimator.src.training.sale_floor_targets import (
    SaleFloorBlendConfig,
    sale_floor_blend_config_from_raw,
    sold_nowcast_s,
)


def _cfg(**kwargs: object) -> SaleFloorBlendConfig:
    raw: dict = {"n_min_trend": 8, "recency_half_life_days": 365.0}
    if kwargs:
        raw["nowcast_stability"] = kwargs
    return sale_floor_blend_config_from_raw(raw, nm_grade_key="Near Mint (NM or M-)")


def test_nowcast_stability_yaml_parsing() -> None:
    cfg = sale_floor_blend_config_from_raw(
        {
            "nowcast_stability": {
                "trend_time_bin": "month",
                "trend_bin_agg": "mean",
                "min_history_span_days": 14.0,
                "tier_a_shrink_lambda": 0.7,
                "tier_b_center": "weighted_median",
            },
        },
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert cfg.trend_time_bin == "month"
    assert cfg.trend_bin_agg == "mean"
    assert cfg.min_history_span_days == 14.0
    assert cfg.tier_a_shrink_lambda == 0.7
    assert cfg.tier_b_center == "weighted_median"


def test_default_config_stable_tier_a_numeric() -> None:
    """Legacy defaults: Tier A path matches prior behavior for a smooth series."""
    t0 = datetime(2020, 1, 1)
    elig = [(t0 + timedelta(days=50 * i), 10.0 * math.exp(0.002 * i)) for i in range(8)]
    t_ref = datetime(2021, 6, 1)
    cfg = _cfg()
    s, tier, n = sold_nowcast_s(elig, t_ref, cfg=cfg)
    assert tier == "A"
    assert n == 8
    assert s is not None and math.isfinite(s) and s > 0


def test_min_history_span_skips_tier_a() -> None:
    t0 = datetime(2020, 1, 1)
    elig = [(t0 + timedelta(days=10 * i), 50.0 + float(i)) for i in range(8)]
    t_ref = datetime(2021, 1, 1)
    cfg = sale_floor_blend_config_from_raw(
        {
            "n_min_trend": 3,
            "nowcast_stability": {"min_history_span_days": 1_000_000.0},
        },
        nm_grade_key="Near Mint (NM or M-)",
    )
    s, tier, n = sold_nowcast_s(elig, t_ref, cfg=cfg)
    assert tier == "B"
    assert n == 8
    assert s is not None and s > 0


def test_tier_a_shrink_lambda_pulls_toward_weighted_median() -> None:
    """lambda=0 replaces Tier A point estimate with recency-weighted median."""
    t0 = datetime(2020, 1, 1)
    elig = [(t0 + timedelta(days=60 * i), 100.0 + float(i)) for i in range(8)]
    t_ref = datetime(2022, 1, 1)
    cfg_full = _cfg(tier_a_shrink_lambda=1.0)
    cfg_shrink0 = _cfg(tier_a_shrink_lambda=0.0)
    s_a, tier_a, _ = sold_nowcast_s(elig, t_ref, cfg=cfg_full)
    s0, tier0, _ = sold_nowcast_s(elig, t_ref, cfg=cfg_shrink0)
    assert tier_a == "A" and tier0 == "A"
    assert s_a is not None and s0 is not None
    ages = [(t_ref - d).total_seconds() / 86400.0 for d, _ in elig]
    H = 365.0
    w = [math.exp(-max(a, 0.0) / H) for a in ages]
    sw = sum(w)
    # Weighted median (same construction as production)
    pairs = sorted(zip([p for _, p in elig], w), key=lambda x: x[0])
    cum = 0.0
    expected_wm = pairs[0][0]
    for p, wt in pairs:
        cum += wt
        expected_wm = p
        if cum >= 0.5 * sw:
            break
    assert s0 == pytest.approx(expected_wm, rel=1e-6)
    assert abs(s_a - s0) > 1e-3


def test_week_bin_smooths_intraweek_jitter_vs_day() -> None:
    """Weekly bin median per ISO week removes outlier swing vs daily Theil–Sen."""
    t0 = datetime(2020, 1, 6, 12, 0, 0)
    elig = []
    for w in range(8):
        base = t0 + timedelta(weeks=w)
        elig.append((base + timedelta(days=1), 10.0))
        elig.append((base + timedelta(days=3), 1000.0))
    t_ref = t0 + timedelta(weeks=20)
    cfg_day = _cfg()
    cfg_week = _cfg(trend_time_bin="week", trend_bin_agg="median")
    s_day, tier_d, _ = sold_nowcast_s(elig, t_ref, cfg=cfg_day)
    s_week, tier_w, _ = sold_nowcast_s(elig, t_ref, cfg=cfg_week)
    assert tier_d == "A"
    assert tier_w == "A"
    assert s_day is not None and s_week is not None
    assert abs(s_week - 505.0) < 25.0
    assert abs(s_day - s_week) > 50.0
