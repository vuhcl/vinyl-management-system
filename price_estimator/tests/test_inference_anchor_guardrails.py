"""Anchor guardrails: gates, trim, blend wiring."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from price_estimator.src.features.vinyliq_features import default_feature_columns
from price_estimator.src.inference.anchor_guardrails import (
    blend_path_anchor_usd,
    guardrails_active,
    is_inflated_ladder,
    prepare_stats_for_inference,
    reference_floor_usd,
    sale_stats_blend_apply,
)
from price_estimator.src.inference.anchor_guardrails_config import AnchorGuardrailsConfig
from price_estimator.src.inference.service import InferenceService
from price_estimator.src.models.fitted_regressor import TARGET_KIND_RESIDUAL_LOG_MEDIAN
from price_estimator.src.training.sale_floor_blend_config import (
    sale_floor_blend_config_from_raw,
    sale_floor_blend_config_with_inference_overrides,
)


def _ladder_12830828() -> str:
    return json.dumps(
        {
            "Mint (M)": {"value": 1900.0, "currency": "USD"},
            "Near Mint (NM or M-)": {"value": 1700.0, "currency": "USD"},
            "Very Good Plus (VG+)": {"value": 1300.0, "currency": "USD"},
        }
    )


def _stats_12830828() -> dict:
    return {
        "release_lowest_price": 649.0,
        "num_for_sale": 5,
        "price_suggestions_json": _ladder_12830828(),
        "sale_stats_median_usd": 600.0,
        "sale_stats_average_usd": 662.91,
    }


def _ladder_456663() -> str:
    return json.dumps(
        {
            "Near Mint (NM or M-)": {"value": 4189.0, "currency": "USD"},
            "Very Good Plus (VG+)": {"value": 3200.0, "currency": "USD"},
            "Good (G)": {"value": 800.0, "currency": "USD"},
        }
    )


def _stats_456663() -> dict:
    return {
        "release_lowest_price": 21.0,
        "num_for_sale": 12,
        "price_suggestions_json": _ladder_456663(),
        "sale_stats_median_usd": 449.99,
        "sale_stats_average_usd": 500.0,
    }


@pytest.fixture
def ag_cfg() -> AnchorGuardrailsConfig:
    return AnchorGuardrailsConfig(enabled=True)


@pytest.fixture
def blend_cfg():
    return sale_floor_blend_config_from_raw({}, nm_grade_key="Near Mint (NM or M-)")


def test_reference_floor_none_when_all_missing(ag_cfg) -> None:
    assert reference_floor_usd({}, ag_cfg, nm_grade_key="Near Mint (NM or M-)") is None
    assert not guardrails_active({}, ag_cfg, nm_grade_key="Near Mint (NM or M-)")


def test_reference_floor_max_12830828(ag_cfg) -> None:
    ref = reference_floor_usd(_stats_12830828(), ag_cfg, nm_grade_key="Near Mint (NM or M-)")
    assert ref == pytest.approx(662.91)


def test_sale_stats_blend_apply_12830828(ag_cfg) -> None:
    stats = _stats_12830828()
    assert is_inflated_ladder(stats, ag_cfg, nm_grade_key="Near Mint (NM or M-)")
    assert sale_stats_blend_apply(stats, ag_cfg, nm_grade_key="Near Mint (NM or M-)")


def test_sale_stats_blend_off_when_ladder_not_inflated(ag_cfg) -> None:
    stats = {
        "release_lowest_price": 100.0,
        "num_for_sale": 12,
        "price_suggestions_json": json.dumps(
            {
                "Near Mint (NM or M-)": {"value": 500.0, "currency": "USD"},
                "Very Good Plus (VG+)": {"value": 400.0, "currency": "USD"},
            }
        ),
        "sale_stats_median_usd": 450.0,
        "sale_stats_average_usd": 460.0,
    }
    assert not sale_stats_blend_apply(stats, ag_cfg, nm_grade_key="Near Mint (NM or M-)")


def test_blend_path_anchor_nm_12830828(ag_cfg, blend_cfg) -> None:
    stats = _stats_12830828()
    out = blend_path_anchor_usd(
        1700.0,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(941.0, abs=5.0)


def test_blend_path_anchor_vg_plus_12830828(ag_cfg, blend_cfg) -> None:
    stats = _stats_12830828()
    out = blend_path_anchor_usd(
        1300.0,
        stats,
        ag_cfg,
        blend_cfg,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(853.0, rel=0.03)


def test_blend_path_anchor_w_base_override_demo(ag_cfg, blend_cfg) -> None:
    stats = _stats_12830828()
    merged = sale_floor_blend_config_with_inference_overrides(
        blend_cfg, {"w_base": 0.6}
    )
    out = blend_path_anchor_usd(
        1700.0,
        stats,
        ag_cfg,
        merged,
        nm_grade_key="Near Mint (NM or M-)",
    )
    assert out == pytest.approx(900.0, rel=0.03)


def test_blend_path_anchor_noop_when_disabled(blend_cfg) -> None:
    cfg = AnchorGuardrailsConfig(enabled=False)
    stats = _stats_12830828()
    assert blend_path_anchor_usd(1700.0, stats, cfg, blend_cfg, nm_grade_key="Near Mint (NM or M-)") == 1700.0


def test_prepare_stats_mint_trim(ag_cfg) -> None:
    stats = {
        "release_lowest_price": 649.0,
        "price_suggestions_json": json.dumps(
            {
                "Mint (M)": {"value": 2500.0, "currency": "USD"},
                "Near Mint (NM or M-)": {"value": 1700.0, "currency": "USD"},
            }
        ),
        "sale_stats_median_usd": 600.0,
    }
    warnings = prepare_stats_for_inference(
        stats, ag_cfg, nm_grade_key="Near Mint (NM or M-)"
    )
    assert "anchor_guardrail_applied" in warnings
    ladder = json.loads(stats["price_suggestions_json"])
    assert ladder["Mint (M)"]["value"] == pytest.approx(1700.0 * 1.15)


class _ResidualZeroPred:
    target_kind = TARGET_KIND_RESIDUAL_LOG_MEDIAN
    backend = "xgboost"
    feature_columns = default_feature_columns()

    def predict_log1p(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=np.float64)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "condition_params.json").write_text(
        json.dumps({"alpha": 0.06, "beta": 0.04, "ref_grade": 8.0})
    )
    return tmp_path


def test_estimate_dual_path_guardrails_anchor_band(model_dir: Path, blend_cfg) -> None:
    mkt = MagicMock()
    mkt.get.return_value = _stats_12830828()
    mkt.upsert = MagicMock()
    fs = MagicMock()
    fs.get.return_value = {"genre": "Rock", "country": "US", "year": 2000}

    svc = InferenceService(
        model_dir=model_dir,
        marketplace_store=mkt,
        feature_store=fs,
        use_price_suggestion_condition_anchor=True,
        anchor_guardrails_cfg=AnchorGuardrailsConfig(enabled=True),
        sale_floor_blend_cfg=blend_cfg,
    )
    svc.redis_cache = MagicMock()
    svc.redis_cache.get.return_value = None
    svc._model = _ResidualZeroPred()

    out = svc.estimate(
        "12830828",
        "Near Mint (NM or M-)",
        "Very Good Plus (VG+)",
        marketplace_client={
            "sale_stats_median_usd": 600.0,
            "sale_stats_average_usd": 662.91,
        },
    )
    assert out["status"] == "ok"
    anchor = float(out["residual_anchor_usd"])
    assert 850.0 <= anchor <= 950.0
    assert out["warnings"].count("anchor_guardrail_applied") == 1
