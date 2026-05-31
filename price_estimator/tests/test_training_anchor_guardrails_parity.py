"""Training ``m_anchor`` applies the same guardrail blend path as inference."""
from __future__ import annotations

import json

import pytest

from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.training.sale_floor_targets import sale_floor_label_diagnostics


def _inflated_mp_row() -> dict:
    return {
        "release_id": "demo",
        "fetched_at": "2024-06-01T00:00:00",
        "release_lowest_price": 649.0,
        "num_for_sale": 5,
        "price_suggestions_json": json.dumps(
            {
                "Mint (M)": {"value": 1900.0, "currency": "USD"},
                "Near Mint (NM or M-)": {"value": 1700.0, "currency": "USD"},
                "Very Good Plus (VG+)": {"value": 1300.0, "currency": "USD"},
            }
        ),
    }


def test_training_m_anchor_guardrail_blend_when_enabled() -> None:
    ag_cfg = AnchorGuardrailsConfig(enabled=True)
    sale_rows = [
        {
            "order_date": "2024-01-01",
            "price_user_usd_approx": 600.0,
            "media_condition": "Near Mint (NM or M-)",
            "sleeve_condition": "Near Mint (NM or M-)",
        },
        {
            "order_date": "2024-02-01",
            "price_user_usd_approx": 725.0,
            "media_condition": "Near Mint (NM or M-)",
            "sleeve_condition": "Near Mint (NM or M-)",
        },
    ]
    fetch = {"status": "ok", "fetched_at": "2024-05-01T00:00:00"}
    sf_cfg = {"sale_condition_policy": "nm_substrings_only", "w_base": 0.55}

    _y, m, _flags, diag = sale_floor_label_diagnostics(
        _inflated_mp_row(),
        sale_rows,
        fetch,
        sf_cfg=sf_cfg,
        nm_grade_key="Near Mint (NM or M-)",
        ag_cfg=ag_cfg,
    )

    assert m is not None
    assert diag.get("reference_floor_training_usd") is not None
    assert diag.get("anchor_guardrail_applied") is True
    assert diag["m_anchor_raw_usd"] == pytest.approx(1700.0)
    assert m == pytest.approx(float(diag["m_anchor_blended_usd"]))
    assert m < 1700.0
    assert 850.0 <= m <= 950.0


def test_training_m_anchor_unchanged_when_guardrails_disabled() -> None:
    ag_cfg = AnchorGuardrailsConfig(enabled=False)
    sale_rows = [
        {
            "order_date": "2024-01-01",
            "price_user_usd_approx": 600.0,
            "media_condition": "Near Mint (NM or M-)",
            "sleeve_condition": "Near Mint (NM or M-)",
        },
    ]
    fetch = {"status": "ok", "fetched_at": "2024-05-01T00:00:00"}
    sf_cfg = {"sale_condition_policy": "nm_substrings_only"}

    _y, m, _flags, diag = sale_floor_label_diagnostics(
        _inflated_mp_row(),
        sale_rows,
        fetch,
        sf_cfg=sf_cfg,
        nm_grade_key="Near Mint (NM or M-)",
        ag_cfg=ag_cfg,
    )

    assert m == pytest.approx(1900.0)
    assert "anchor_guardrail_applied" not in diag
