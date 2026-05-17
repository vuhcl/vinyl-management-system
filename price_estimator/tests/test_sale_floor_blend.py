"""Unit tests for shared sale-floor log-blend."""
from __future__ import annotations

import pytest

from price_estimator.src.sale_floor.blend import (
    blend_weight_w_eff,
    blend_weight_w_eff_for_anchor,
    sale_floor_blend_y,
    sale_floor_blend_y_for_anchor,
)
from price_estimator.src.training.sale_floor_blend_config import (
    sale_floor_blend_config_from_raw,
    sale_floor_blend_config_with_inference_overrides,
)


@pytest.fixture
def blend_cfg():
    return sale_floor_blend_config_from_raw(
        {"w_base": 0.55, "w_min": 0.2, "w_max": 0.9},
        nm_grade_key="Near Mint (NM or M-)",
    )


def test_blend_weight_tier_a_defaults(blend_cfg) -> None:
    w = blend_weight_w_eff(s=631.0, lo=1700.0, tier="A", cfg=blend_cfg)
    assert 0.2 <= w <= 0.9
    assert w == pytest.approx(0.49, abs=0.02)


def test_sale_floor_blend_y_12830828_nm_path(blend_cfg) -> None:
    y = sale_floor_blend_y(663.0, 1700.0, "A", cfg=blend_cfg)
    assert y is not None
    assert y == pytest.approx(1055.0, rel=0.02)


def test_sale_floor_blend_y_no_s_returns_none(blend_cfg) -> None:
    assert sale_floor_blend_y(None, 100.0, "A", cfg=blend_cfg) == 100.0
    assert sale_floor_blend_y(None, None, "A", cfg=blend_cfg) is None


def test_anchor_blend_weight_higher_when_rung_above_reference(blend_cfg) -> None:
    w_train = blend_weight_w_eff(s=663.0, lo=1700.0, tier="A", cfg=blend_cfg)
    w_anchor = blend_weight_w_eff_for_anchor(
        reference_usd=663.0, rung_usd=1700.0, tier="A", cfg=blend_cfg
    )
    assert w_anchor > w_train


def test_sale_floor_blend_y_for_anchor_12830828_nm_path(blend_cfg) -> None:
    y = sale_floor_blend_y_for_anchor(663.0, 1700.0, "A", cfg=blend_cfg)
    assert y is not None
    assert y == pytest.approx(944.0, rel=0.02)


def test_inference_blend_override_w_base(blend_cfg) -> None:
    merged = sale_floor_blend_config_with_inference_overrides(
        blend_cfg, {"w_base": 0.6}
    )
    y = sale_floor_blend_y_for_anchor(663.0, 1700.0, "A", cfg=merged)
    assert y is not None
    assert y == pytest.approx(900.0, rel=0.02)
