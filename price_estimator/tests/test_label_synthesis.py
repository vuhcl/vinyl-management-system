"""Training target synthesis from marketplace median/lowest."""
from __future__ import annotations

import pytest

from price_estimator.src.training.label_synthesis import (
    synthesize_training_price,
    training_label_config_from_vinyliq,
)


def test_median_mode():
    assert synthesize_training_price(20.0, 10.0, mode="median") == 20.0


def test_blend():
    y = synthesize_training_price(20.0, 10.0, mode="blend", blend_median_weight=0.7)
    assert y == pytest.approx(17.0)


def test_blend_missing_lowest_falls_back_to_median():
    assert synthesize_training_price(20.0, None, mode="blend", blend_median_weight=0.5) == 20.0


def test_geometric_mean():
    y = synthesize_training_price(16.0, 4.0, mode="geometric_mean")
    assert y == pytest.approx(8.0)


def test_spread_signal_mid_spread():
    # c=0.5 → y = 0.5*20 + 0.5*10 = 15 (same midpoint as old symmetric mix)
    y = synthesize_training_price(20.0, 10.0, mode="spread_signal")
    assert y == pytest.approx(15.0)


def test_spread_signal_tight_market():
    y = synthesize_training_price(20.0, 20.0, mode="spread_signal")
    assert y == pytest.approx(20.0)


def test_spread_signal_wide_pulls_toward_lowest_not_median_raw():
    # No robustness kwargs: c=0.2 → y = 0.2*20 + 0.8*4 = 7.2 (outlier-sensitive).
    y = synthesize_training_price(20.0, 4.0, mode="spread_signal")
    assert y == pytest.approx(7.2)


def test_spread_signal_wide_with_robust_defaults():
    # lo_eff=max(4,0.35*20)=7; raw_c=0.35; c=max(0.25,0.35)=0.35
    # y = 0.35*20 + 0.65*7 = 11.55
    y = synthesize_training_price(
        20.0,
        4.0,
        mode="spread_signal",
        spread_lowest_floor_ratio=0.35,
        spread_min_median_weight=0.25,
    )
    assert y == pytest.approx(11.55)


def test_spread_signal_lowest_above_median_uses_median():
    y = synthesize_training_price(10.0, 25.0, mode="spread_signal")
    assert y == pytest.approx(10.0)


def test_spread_signal_missing_lowest():
    assert synthesize_training_price(20.0, None, mode="spread_signal") == 20.0


def test_spread_alias():
    assert synthesize_training_price(20.0, 10.0, mode="spread") == pytest.approx(15.0)


def test_invalid_mode():
    with pytest.raises(ValueError):
        synthesize_training_price(10.0, 5.0, mode="nope")


def test_training_label_config_from_vinyliq_defaults():
    assert training_label_config_from_vinyliq({}) == {
        "mode": "median",
        "blend_median_weight": 0.7,
        "spread_lowest_floor_ratio": None,
        "spread_min_median_weight": None,
        "spread_num_for_sale_reference": None,
    }


def test_training_label_config_from_vinyliq_nested():
    cfg = {"training_label": {"mode": "blend", "blend_median_weight": 0.6}}
    assert training_label_config_from_vinyliq(cfg) == {
        "mode": "blend",
        "blend_median_weight": 0.6,
        "spread_lowest_floor_ratio": None,
        "spread_min_median_weight": None,
        "spread_num_for_sale_reference": None,
    }


def test_training_label_config_spread_includes_robust_defaults():
    cfg = {"training_label": {"mode": "spread_signal"}}
    d = training_label_config_from_vinyliq(cfg)
    assert d["mode"] == "spread_signal"
    assert d["spread_lowest_floor_ratio"] == pytest.approx(0.35)
    assert d["spread_min_median_weight"] == pytest.approx(0.25)
    assert d["spread_num_for_sale_reference"] == pytest.approx(20.0)


def test_training_label_config_spread_explicit_null_disables_robust():
    cfg = {
        "training_label": {
            "mode": "spread_signal",
            "spread_lowest_floor_ratio": None,
            "spread_min_median_weight": None,
        }
    }
    d = training_label_config_from_vinyliq(cfg)
    assert d["spread_lowest_floor_ratio"] is None
    assert d["spread_min_median_weight"] is None
    assert d["spread_num_for_sale_reference"] == pytest.approx(20.0)


def test_spread_signal_listing_depth_wide_spread_partial_median_pull():
    # lo_eff=7, gap=0.65; y_spread=11.55; n=10, ref=20 → t=0.5 → pull=0.325
    y = synthesize_training_price(
        20.0,
        4.0,
        mode="spread_signal",
        spread_lowest_floor_ratio=0.35,
        spread_min_median_weight=0.25,
        num_for_sale=10,
        spread_num_for_sale_reference=20.0,
    )
    assert y == pytest.approx(14.29625)


def test_spread_signal_listing_depth_wide_spread_high_n_pulls_median():
    # gap=0.65, t=1 → pull=0.65 → y = 0.35*11.55 + 0.65*20
    y = synthesize_training_price(
        20.0,
        4.0,
        mode="spread_signal",
        spread_lowest_floor_ratio=0.35,
        spread_min_median_weight=0.25,
        num_for_sale=25,
        spread_num_for_sale_reference=20.0,
    )
    assert y == pytest.approx(17.0425)


def test_spread_signal_listing_depth_low_n_wide_spread_keeps_floor_signal():
    # n=0 → t=0 → pull=0 → full y_spread (few listings: do not force median)
    y = synthesize_training_price(
        20.0,
        4.0,
        mode="spread_signal",
        spread_lowest_floor_ratio=0.35,
        spread_min_median_weight=0.25,
        num_for_sale=0,
        spread_num_for_sale_reference=20.0,
    )
    assert y == pytest.approx(11.55)


def test_spread_signal_listing_depth_tight_book_no_pull_despite_high_n():
    # gap=0 regardless of n
    y = synthesize_training_price(
        20.0,
        20.0,
        mode="spread_signal",
        spread_lowest_floor_ratio=0.35,
        spread_min_median_weight=0.25,
        num_for_sale=500,
        spread_num_for_sale_reference=20.0,
    )
    assert y == pytest.approx(20.0)


def test_spread_signal_listing_reference_none_skips_depth_blend():
    y = synthesize_training_price(
        20.0,
        4.0,
        mode="spread_signal",
        spread_lowest_floor_ratio=0.35,
        spread_min_median_weight=0.25,
        num_for_sale=0,
        spread_num_for_sale_reference=None,
    )
    assert y == pytest.approx(11.55)


def test_training_label_config_spread_null_reference_disables_listing_depth():
    cfg = {
        "training_label": {
            "mode": "spread_signal",
            "spread_num_for_sale_reference": None,
        }
    }
    d = training_label_config_from_vinyliq(cfg)
    assert d["spread_num_for_sale_reference"] is None
