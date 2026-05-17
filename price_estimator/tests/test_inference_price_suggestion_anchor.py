"""Residual serving: ladder-rung anchors for requested media/sleeve grades."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from price_estimator.src.features.vinyliq_features import (
    default_feature_columns,
)
from price_estimator.src.inference.service import InferenceService
from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
)
from price_estimator.src.training.sale_floor_targets import (
    inference_price_suggestion_anchor_usd_for_side,
    inference_price_suggestion_condition_anchor_usd,
)


def _ladder_stub() -> str:
    return json.dumps(
        {
            "Near Mint (NM or M-)": {"value": 100.0, "currency": "USD"},
            "Good (G)": {"value": 55.0, "currency": "USD"},
        }
    )


def _fake_mkt_ps(*, lad: str | None, num_sale: int = 10) -> MagicMock:
    m = MagicMock()
    payload: dict[str, object] = {
        "release_lowest_price": 500.0,
        "num_for_sale": num_sale,
    }
    if lad is not None:
        payload["price_suggestions_json"] = lad
    m.get.return_value = payload
    m.upsert = MagicMock()
    return m


def _fake_fs() -> MagicMock:
    fs = MagicMock()
    fs.get.return_value = {
        "genre": "Rock",
        "country": "US",
        "year": 2000,
    }
    return fs


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


def test_price_suggestion_anchor_usd_media_role() -> None:
    row = {"price_suggestions_json": _ladder_stub()}
    v = inference_price_suggestion_anchor_usd_for_side(
        row,
        role="media",
        media_condition="Good (G)",
        sleeve_condition="Near Mint (NM or M-)",
    )
    assert v == pytest.approx(55.0)


def test_price_suggestion_anchor_usd_sleeve_role() -> None:
    row = {"price_suggestions_json": _ladder_stub()}
    v = inference_price_suggestion_anchor_usd_for_side(
        row,
        role="sleeve",
        media_condition="Good (G)",
        sleeve_condition="Near Mint (NM or M-)",
    )
    assert v == pytest.approx(100.0)


def test_inference_price_suggestion_anchor_min_of_media_and_sleeve() -> None:
    row = {"price_suggestions_json": _ladder_stub()}
    v = inference_price_suggestion_condition_anchor_usd(
        row,
        media_condition="Good (G)",
        sleeve_condition="Near Mint (NM or M-)",
    )
    assert v == pytest.approx(55.0)


def test_inference_price_suggestion_anchor_single_side_maps() -> None:
    row = {"price_suggestions_json": _ladder_stub()}
    v = inference_price_suggestion_condition_anchor_usd(
        row,
        media_condition="Good (G)",
        sleeve_condition=None,
    )
    assert v == pytest.approx(55.0)


def test_residual_mode_averages_media_and_sleeve_ps_paths(model_dir: Path) -> None:
    svc = InferenceService(
        model_dir=model_dir,
        marketplace_store=_fake_mkt_ps(lad=_ladder_stub()),
        feature_store=_fake_fs(),
        use_price_suggestion_condition_anchor=True,
    )
    svc.redis_cache = MagicMock()
    svc.redis_cache.get.return_value = None
    svc._model = _ResidualZeroPred()

    out = svc.estimate("1", "Good (G)", "Near Mint (NM or M-)")

    assert out["status"] == "ok"
    assert out["residual_anchor_usd"] == pytest.approx(77.5)
    assert out["estimated_price"] == pytest.approx(77.5)


def test_residual_legacy_anchor_when_ps_flag_off(model_dir: Path) -> None:
    svc = InferenceService(
        model_dir=model_dir,
        marketplace_store=_fake_mkt_ps(lad=_ladder_stub()),
        feature_store=_fake_fs(),
        use_price_suggestion_condition_anchor=False,
    )
    svc.redis_cache = MagicMock()
    svc.redis_cache.get.return_value = None
    svc._model = _ResidualZeroPred()

    out = svc.estimate("1", "Good (G)", "Near Mint (NM or M-)")

    assert out["status"] == "ok"
    assert out["residual_anchor_usd"] == pytest.approx(100.0)
