"""Unit tests for shared sale-floor log-blend."""
from __future__ import annotations

import pytest

from price_estimator.src.sale_floor.blend import blend_weight_w_eff, sale_floor_blend_y
from price_estimator.src.training.sale_floor_blend_config import (
    sale_floor_blend_config_from_raw,
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
