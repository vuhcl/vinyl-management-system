"""Unit tests for ratio-based PS ladder blend strength."""
from __future__ import annotations

import json

import pytest

from price_estimator.src.inference.anchor_guardrails import (
    blend_path_anchor_usd,
    blend_strength_for_grade,
)
from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.sale_floor.ratio_guardrails import (
    compute_ratio_sale_ladder,
    listing_credible_vs_sale_low,
)
from price_estimator.src.training.sale_floor_blend_config import (
    sale_floor_blend_config_from_raw,
)


@pytest.fixture
def ag_cfg() -> AnchorGuardrailsConfig:
    return AnchorGuardrailsConfig(enabled=True)


@pytest.fixture
def blend_cfg():
    return sale_floor_blend_config_from_raw({"w_base": 0.55}, nm_grade_key="Near Mint (NM or M-)")


def _ladder(nm: float) -> str:
    return json.dumps(
        {"Near Mint (NM or M-)": {"value": nm, "currency": "USD"}},
        separators=(",", ":"),
    )


def test_listing_excluded_when_above_sale_high(ag_cfg) -> None:
    stats = {
        "release_lowest_price": 5000.0,
        "sale_stats_low_usd": 26.67,
        "sale_stats_high_usd": 2666.67,
    }
    assert not listing_credible_vs_sale_low(stats, ag_cfg)


def test_listing_excluded_when_below_sale_low(ag_cfg) -> None:
    stats = {
        "release_lowest_price": 75.0,
        "sale_stats_low_usd": 91.0,
        "sale_stats_high_usd": 4126.70,
    }
    assert not listing_credible_vs_sale_low(stats, ag_cfg)


def test_ex1_down_blend_calibration(ag_cfg, blend_cfg) -> None:
    stats = {
        "release_lowest_price": 92.0,
        "n_sales": 5,
        "price_suggestions_json": _ladder(173.86),
        "sale_stats_low_usd": 45.93,
        "sale_stats_median_usd": 91.01,
        "sale_stats_average_usd": 102.42,
        "sale_stats_high_usd": 204.55,
    }
    strength, diag = blend_strength_for_grade(stats, ag_cfg, grade_rung_usd=173.86)
    assert strength == pytest.approx(0.98, abs=0.05)
    assert diag.blend_direction == "down"
    out = blend_path_anchor_usd(
        173.86,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(126.0, abs=2.0)


def test_ex2_left_skew_skip_down_blend(ag_cfg, blend_cfg) -> None:
    stats = {
        "release_lowest_price": 350.0,
        "n_sales": 5,
        "price_suggestions_json": _ladder(488.75),
        "sale_stats_low_usd": 63.95,
        "sale_stats_median_usd": 315.43,
        "sale_stats_average_usd": 293.03,
        "sale_stats_high_usd": 678.57,
    }
    diag = compute_ratio_sale_ladder(stats, ag_cfg, grade_rung_usd=488.75)
    assert diag.left_skew is True
    assert diag.skip_down_blend is True
    assert diag.blend_strength == pytest.approx(0.0, abs=1e-6)
    out = blend_path_anchor_usd(
        488.75,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(488.75, abs=2.0)


def test_ex3_no_blend_when_ladder_below_rsale(ag_cfg, blend_cfg) -> None:
    stats = {
        "release_lowest_price": 5000.0,
        "n_sales": 5,
        "price_suggestions_json": _ladder(552.50),
        "sale_stats_low_usd": 26.67,
        "sale_stats_median_usd": 250.0,
        "sale_stats_average_usd": 532.98,
        "sale_stats_high_usd": 2666.67,
    }
    diag = compute_ratio_sale_ladder(stats, ag_cfg, grade_rung_usd=552.50)
    assert not diag.listing_credible_vs_sale_low
    assert diag.blend_strength == pytest.approx(0.0, abs=1e-6)
    out = blend_path_anchor_usd(
        552.50,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(552.50, abs=2.0)


def test_red_ts_full_down_blend(ag_cfg, blend_cfg) -> None:
    stats = {
        "release_lowest_price": 400.0,
        "n_sales": 5,
        "price_suggestions_json": _ladder(1700.0),
        "sale_stats_low_usd": 400.0,
        "sale_stats_median_usd": 600.0,
        "sale_stats_average_usd": 660.0,
        "sale_stats_high_usd": 2000.0,
    }
    strength, diag = blend_strength_for_grade(stats, ag_cfg, grade_rung_usd=1700.0)
    assert strength == pytest.approx(1.0, abs=0.05)
    assert diag.R_sale == pytest.approx(1.65, rel=0.02)
    out = blend_path_anchor_usd(
        1700.0,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(941.0, abs=2.0)


def test_red_high_invariant_rsale(ag_cfg) -> None:
    base = {
        "release_lowest_price": 400.0,
        "n_sales": 5,
        "sale_stats_low_usd": 400.0,
        "sale_stats_median_usd": 600.0,
        "sale_stats_average_usd": 660.0,
    }
    low_high = {**base, "sale_stats_high_usd": 1300.0}
    high_high = {**base, "sale_stats_high_usd": 2000.0}
    d1 = compute_ratio_sale_ladder(low_high, ag_cfg, grade_rung_usd=1700.0)
    d2 = compute_ratio_sale_ladder(high_high, ag_cfg, grade_rung_usd=1700.0)
    assert d1.R_sale == pytest.approx(d2.R_sale, rel=1e-6)


def test_white_album_no_blend(ag_cfg, blend_cfg) -> None:
    stats = {
        "release_lowest_price": 75.0,
        "n_sales": 10,
        "price_suggestions_json": _ladder(4133.20),
        "sale_stats_low_usd": 91.0,
        "sale_stats_median_usd": 449.99,
        "sale_stats_average_usd": 801.30,
        "sale_stats_high_usd": 4126.70,
    }
    strength, diag = blend_strength_for_grade(stats, ag_cfg, grade_rung_usd=4133.20)
    assert strength == pytest.approx(0.0, abs=1e-6)
    assert diag.R_sale == pytest.approx(8.81, rel=0.05)
    out = blend_path_anchor_usd(
        4133.20,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(4133.20, abs=2.0)


def test_buyer_legacy_partial_blend_when_avg_missing(ag_cfg, blend_cfg) -> None:
    stats = {
        "release_lowest_price": 400.0,
        "n_sales": 5,
        "price_suggestions_json": _ladder(1700.0),
        "sale_stats_low_usd": 400.0,
        "sale_stats_median_usd": 600.0,
        "sale_stats_high_usd": 2000.0,
    }
    strength, diag = blend_strength_for_grade(stats, ag_cfg, grade_rung_usd=1700.0)
    assert diag.ratio_blend_fallback is True
    assert diag.sale_stats_average_missing is True
    assert strength == pytest.approx(0.34, abs=0.05)
    out = blend_path_anchor_usd(
        1700.0,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(1366.0, abs=30.0)
