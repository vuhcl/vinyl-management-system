"""Tests for training vs inference reference floor + gate outcomes."""
from __future__ import annotations

import json

import pytest

from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.sale_floor.reference_floor import (
    gate_outcomes_for_ref,
    reference_floor_inference_usd,
    reference_floor_training_usd,
    sale_stats_from_history,
)
from datetime import datetime


def _inflated_ladder_stats() -> dict:
    return {
        "release_lowest_price": 649.0,
        "num_for_sale": 5,
        "price_suggestions_json": json.dumps(
            {
                "Mint (M)": {"value": 1900.0, "currency": "USD"},
                "Near Mint (NM or M-)": {"value": 1700.0, "currency": "USD"},
                "Very Good Plus (VG+)": {"value": 1300.0, "currency": "USD"},
            }
        ),
        "sale_stats_median_usd": 600.0,
        "sale_stats_average_usd": 662.91,
    }


@pytest.fixture
def ag_cfg() -> AnchorGuardrailsConfig:
    return AnchorGuardrailsConfig(enabled=True)


def test_reference_floor_inference_uses_sale_stats(ag_cfg) -> None:
    ref = reference_floor_inference_usd(
        _inflated_ladder_stats(), ag_cfg, nm_grade_key="Near Mint (NM or M-)"
    )
    assert ref == pytest.approx(662.91)


def test_gate_outcomes_blend_when_inflated(ag_cfg) -> None:
    stats = _inflated_ladder_stats()
    ref = 662.91
    out = gate_outcomes_for_ref(ref, stats, ag_cfg, nm_grade_key="Near Mint (NM or M-)")
    assert out["is_inflated_ladder"] is True
    assert out["sale_stats_blend_apply"] is True
    assert out["ratio_mx_ref"] == pytest.approx(1900.0 / 662.91, rel=0.01)


def test_gate_outcomes_blend_even_when_mx_ref_very_high(ag_cfg) -> None:
    stats = {
        "release_lowest_price": 10.0,
        "price_suggestions_json": json.dumps(
            {"Near Mint (NM or M-)": {"value": 600.0, "currency": "USD"}}
        ),
    }
    ref = 50.0
    out = gate_outcomes_for_ref(ref, stats, ag_cfg, nm_grade_key="Near Mint (NM or M-)")
    assert out["is_inflated_ladder"] is True
    assert out["sale_stats_blend_apply"] is True


def test_sale_stats_from_history_all_sales_before_t_ref() -> None:
    t_ref = datetime(2024, 6, 1)
    rows = [
        {"order_date": "2024-01-01", "price_user_usd_approx": 100.0},
        {"order_date": "2024-03-01", "price_user_usd_approx": 200.0},
        {"order_date": "2024-07-01", "price_user_usd_approx": 999.0},
    ]
    stats = sale_stats_from_history(rows, t_ref)
    assert stats["n_sales"] == 2
    assert stats["sale_stats_median_usd"] == pytest.approx(150.0)
    assert stats["sale_stats_average_usd"] == pytest.approx(150.0)


def test_reference_floor_training_from_history_quartiles(ag_cfg) -> None:
    mp_row = {
        "release_id": "1",
        "fetched_at": "2024-06-01T00:00:00",
        "release_lowest_price": 80.0,
        "price_suggestions_json": json.dumps(
            {"Near Mint (NM or M-)": {"value": 120.0, "currency": "USD"}}
        ),
    }
    sale_rows = [
        {"order_date": "2024-01-01", "price_user_usd_approx": 100.0},
        {"order_date": "2024-02-01", "price_user_usd_approx": 200.0},
    ]
    fetch = {"status": "ok", "fetched_at": "2024-05-01T00:00:00"}
    ref, diag = reference_floor_training_usd(
        mp_row,
        sale_rows,
        fetch,
        nm_grade_key="Near Mint (NM or M-)",
        ag_cfg=ag_cfg,
    )
    assert ref == pytest.approx(150.0)
    assert diag["sale_stats_median_usd"] == pytest.approx(150.0)
    assert diag["credible_listing_usd"] == pytest.approx(80.0)


def test_reference_floor_training_from_listing_only(ag_cfg) -> None:
    mp_row = {
        "release_id": "1",
        "fetched_at": "2024-01-01T00:00:00",
        "release_lowest_price": 100.0,
        "price_suggestions_json": json.dumps(
            {"Near Mint (NM or M-)": {"value": 120.0, "currency": "USD"}}
        ),
    }
    ref, diag = reference_floor_training_usd(
        mp_row,
        [],
        None,
        nm_grade_key="Near Mint (NM or M-)",
        ag_cfg=ag_cfg,
    )
    assert ref == pytest.approx(100.0)
    assert diag["credible_listing_usd"] == pytest.approx(100.0)
    assert diag["n_sales"] == 0
