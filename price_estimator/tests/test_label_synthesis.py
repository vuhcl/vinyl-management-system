"""Training target helpers (sale-floor training + auxiliary dollar_target paths)."""
from __future__ import annotations

from pathlib import Path

import pytest

from price_estimator.src.training.label_synthesis import (
    dollar_target_and_residual_anchor_from_marketplace_row,
    parse_price_suggestion_value,
    training_label_config_from_vinyliq,
)
from price_estimator.src.training.train_vinyliq import load_training_frame


def test_training_label_config_from_vinyliq_defaults():
    assert training_label_config_from_vinyliq({}) == {
        "mode": "sale_floor_blend",
        "price_suggestion_grade": "Near Mint (NM or M-)",
        "price_suggestion_fallback_lowest": True,
        "sale_floor_blend": {},
    }


def test_training_label_config_from_vinyliq_nested():
    cfg = {
        "training_label": {
            "mode": "sale_floor",
            "price_suggestion_grade": "Very Good Plus (VG+)",
            "sale_floor_blend": {"w_base": 0.5},
        }
    }
    assert training_label_config_from_vinyliq(cfg) == {
        "mode": "sale_floor",
        "price_suggestion_grade": "Very Good Plus (VG+)",
        "price_suggestion_fallback_lowest": True,
        "sale_floor_blend": {"w_base": 0.5},
    }


def test_parse_price_suggestion_value():
    raw = '{"Near Mint (NM or M-)": {"currency": "USD", "value": 12.5}}'
    assert parse_price_suggestion_value(raw, "Near Mint (NM or M-)") == pytest.approx(
        12.5
    )


def test_dollar_target_release_lowest_uses_release_lowest_price():
    row = {
        "release_lowest_price": 18.0,
        "num_for_sale": 1,
    }
    tl = {"mode": "release_lowest", "price_suggestion_grade": "Near Mint (NM or M-)"}
    y, m = dollar_target_and_residual_anchor_from_marketplace_row(row, tl)
    assert y == pytest.approx(18.0)
    assert m == pytest.approx(18.0)


def test_dollar_target_price_suggestion():
    row = {
        "price_suggestions_json": (
            '{"Near Mint (NM or M-)": {"currency": "USD", "value": 30.0}}'
        ),
        "release_lowest_price": 20.0,
        "num_for_sale": 2,
    }
    tl = {
        "mode": "price_suggestion",
        "price_suggestion_grade": "Near Mint (NM or M-)",
        "price_suggestion_fallback_lowest": True,
    }
    y, m = dollar_target_and_residual_anchor_from_marketplace_row(row, tl)
    assert y == pytest.approx(30.0)
    assert m == pytest.approx(20.0)


def test_dollar_target_retired_median_mode_raises():
    row = {"release_lowest_price": 10.0}
    tl = {"mode": "median"}
    with pytest.raises(ValueError, match="retired"):
        dollar_target_and_residual_anchor_from_marketplace_row(row, tl)


def test_dollar_target_sale_floor_mode_raises():
    row = {"release_lowest_price": 10.0}
    tl = {"mode": "sale_floor_blend"}
    with pytest.raises(ValueError, match="load_training_frame"):
        dollar_target_and_residual_anchor_from_marketplace_row(row, tl)


def test_load_training_frame_rejects_non_sale_floor_modes():
    with pytest.raises(ValueError, match="Unsupported"):
        load_training_frame(
            Path("/no/such/marketplace.sqlite"),
            Path("/no/such/feature_store.sqlite"),
            training_label={"mode": "spread_signal"},
        )
