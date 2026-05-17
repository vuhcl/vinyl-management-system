"""Inference guardrails G2–G7 (service + shared constants)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from price_estimator.src.features import vinyliq_features as vf
from price_estimator.src.features.vinyliq_features import (
    MAX_LOG_PRICE,
    MAX_PRICE_USD,
    clamp_ordinals_for_inference,
    default_feature_columns,
)
from price_estimator.src.inference.service import InferenceService
from price_estimator.src.models.fitted_regressor import TARGET_KIND_DOLLAR_LOG1P


def _fake_marketplace(num_for_sale: int = 1) -> MagicMock:
    m = MagicMock()
    m.get.return_value = {
        "release_lowest_price": 25.0,
        "num_for_sale": num_for_sale,
    }
    m.upsert = MagicMock()
    return m


def _fake_feature_store(year: float = 2000.0) -> MagicMock:
    fs = MagicMock()
    fs.get.return_value = {
        "genre": "Rock",
        "country": "US",
        "year": year,
    }
    return fs


class _FakeRegressor:
    target_kind = TARGET_KIND_DOLLAR_LOG1P
    backend = "xgboost"

    def __init__(self, log1p_values: np.ndarray) -> None:
        self._vals = np.asarray(log1p_values, dtype=np.float64)
        self.feature_columns = default_feature_columns()
        self._i = 0

    def predict_log1p(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        idx = min(self._i, len(self._vals) - 1)
        out = np.full(n, float(self._vals[idx]))
        self._i += n
        return out


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "condition_params.json").write_text(
        json.dumps({"alpha": 0.06, "beta": 0.04, "ref_grade": 8.0})
    )
    return tmp_path


@pytest.fixture
def svc(model_dir: Path) -> InferenceService:
    s = InferenceService(
        model_dir=model_dir,
        marketplace_store=_fake_marketplace(num_for_sale=1),
        feature_store=_fake_feature_store(),
    )
    s.redis_cache = MagicMock()
    s.redis_cache.get.return_value = None
    return s


def test_clamp_ordinals_sentinels_to_ladder() -> None:
    mo, so = clamp_ordinals_for_inference(-1.0, -2.0)
    assert mo == 1.0 and so == 1.0
    mo2, so2 = clamp_ordinals_for_inference(10.0, 0.5)
    assert mo2 == 8.0 and so2 == 1.0


def test_max_log_price_matches_cap() -> None:
    assert MAX_LOG_PRICE == pytest.approx(math.log1p(MAX_PRICE_USD))


def test_estimate_warns_low_market_depth(svc: InferenceService) -> None:
    svc._model = _FakeRegressor(np.array([math.log1p(40.0)]))
    out = svc.estimate("1", "Near Mint (NM or M-)", "Near Mint (NM or M-)")
    assert out["status"] == "ok"
    assert out["num_for_sale"] == 1
    assert "low_market_depth" in out["warnings"]


def test_estimate_no_low_depth_when_enough_listings(svc: InferenceService) -> None:
    svc.marketplace = _fake_marketplace(num_for_sale=5)
    svc._model = _FakeRegressor(np.array([math.log1p(40.0)]))
    out = svc.estimate("1", "Near Mint (NM or M-)", "Near Mint (NM or M-)")
    assert out["warnings"] == []


def test_price_floor(svc: InferenceService) -> None:
    svc._model = _FakeRegressor(np.array([-999.0]))
    out = svc.estimate("1", "Near Mint (NM or M-)", "Near Mint (NM or M-)")
    assert out["estimated_price"] == pytest.approx(0.5)


def test_log_price_ceiling(svc: InferenceService) -> None:
    svc._model = _FakeRegressor(np.array([MAX_LOG_PRICE + 5.0]))
    out = svc.estimate("1", "Near Mint (NM or M-)", "Near Mint (NM or M-)")
    assert out["estimated_price"] == pytest.approx(MAX_PRICE_USD, rel=1e-9)


def test_release_year_clamped(svc: InferenceService) -> None:
    svc.features = _fake_feature_store(year=9999.0)
    svc._model = _FakeRegressor(np.array([math.log1p(40.0)]))
    out = svc.estimate("1", "Very Good (VG)", "Near Mint (NM or M-)")
    assert out["status"] == "ok"
    assert math.isfinite(float(out["estimated_price"]))


def test_row_dict_uses_nm_strings_for_model_inputs(
    svc: InferenceService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture_row(*args: object, **kwargs: object) -> dict[str, float]:
        out = vf.row_dict_for_inference(*args, **kwargs)
        captured["media_passed"] = args[1]
        captured["sleeve_passed"] = args[2]
        return out

    monkeypatch.setattr(
        "price_estimator.src.inference.service.row_dict_for_inference",
        _capture_row,
    )
    svc._model = _FakeRegressor(np.array([math.log1p(40.0)]))
    svc.estimate("1", "Poor (P)", "Good (G)")
    assert captured["media_passed"] == "Near Mint (NM or M-)"
    assert captured["sleeve_passed"] == "Near Mint (NM or M-)"
