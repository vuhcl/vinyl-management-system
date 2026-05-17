"""Integration-style tests: calibration + quantile resolution on a temp model_dir."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from price_estimator.src.features.vinyliq_features import default_feature_columns
from price_estimator.src.inference.service import InferenceService
from price_estimator.src.models.fitted_regressor import TARGET_KIND_DOLLAR_LOG1P
from price_estimator.src.models.regressor_constants import (
    CONFIDENCE_CALIBRATION_FILE,
    REGRESSOR_Q_HIGH_FILE,
    REGRESSOR_Q_LOW_FILE,
)


def _fake_marketplace() -> MagicMock:
    m = MagicMock()
    m.get.return_value = {
        "release_lowest_price": 25.0,
        "num_for_sale": 5,
    }
    m.upsert = MagicMock()
    return m


def _fake_feature_store() -> MagicMock:
    fs = MagicMock()
    fs.get.return_value = {
        "genre": "Rock",
        "country": "US",
        "year": 2000.0,
    }
    return fs


class _FakeDollarRegressor:
    target_kind = TARGET_KIND_DOLLAR_LOG1P
    backend = "xgboost"

    def __init__(self, log1p_val: float) -> None:
        self._v = float(log1p_val)
        self.feature_columns = default_feature_columns()

    def predict_log1p(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._v, dtype=np.float64)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "condition_params.json").write_text(
        json.dumps({"alpha": 0.06, "beta": 0.04, "ref_grade": 8.0})
    )
    return tmp_path


def test_estimate_uses_calibration_half_width(
    model_dir: Path,
) -> None:
    (model_dir / CONFIDENCE_CALIBRATION_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fallback_half_width_usd": 6.0,
                "holdout": {"half_width_usd": 6.0},
            }
        )
    )
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"schema_version": 2, "backend": "xgboost", "target_kind": "dollar_log1p"})
    )

    svc = InferenceService(
        model_dir=model_dir,
        marketplace_store=_fake_marketplace(),
        feature_store=_fake_feature_store(),
    )
    svc.redis_cache = MagicMock()
    svc.redis_cache.get.return_value = None
    svc._model = _FakeDollarRegressor(math.log1p(42.0))

    out = svc.estimate("1", "Near Mint (NM or M-)", "Near Mint (NM or M-)")
    assert out["status"] == "ok"
    est = float(out["estimated_price"])
    lo, hi = out["confidence_interval"]
    assert lo <= est <= hi
    assert hi - lo >= 11.0


def test_estimate_quantile_interval_ordering(model_dir: Path) -> None:
    lp_point = math.log1p(40.0)
    lp_lo = lp_point - 0.05
    lp_hi = lp_point + 0.05

    joblib.dump(
        DummyRegressor(strategy="constant", constant=lp_lo),
        model_dir / REGRESSOR_Q_LOW_FILE,
    )
    joblib.dump(
        DummyRegressor(strategy="constant", constant=lp_hi),
        model_dir / REGRESSOR_Q_HIGH_FILE,
    )
    (model_dir / CONFIDENCE_CALIBRATION_FILE).write_text(
        json.dumps({"fallback_half_width_usd": 50.0})
    )
    (model_dir / "model_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "xgboost",
                "target_kind": "dollar_log1p",
                "quantile_intervals": {
                    "enabled": True,
                    "lower": REGRESSOR_Q_LOW_FILE,
                    "upper": REGRESSOR_Q_HIGH_FILE,
                    "lower_alpha": 0.1,
                    "upper_alpha": 0.9,
                },
            }
        )
    )

    svc = InferenceService(
        model_dir=model_dir,
        marketplace_store=_fake_marketplace(),
        feature_store=_fake_feature_store(),
    )
    svc.redis_cache = MagicMock()
    svc.redis_cache.get.return_value = None
    svc._model = _FakeDollarRegressor(lp_point)

    out = svc.estimate("1", "Near Mint (NM or M-)", "Near Mint (NM or M-)")
    est = float(out["estimated_price"])
    lo, hi = out["confidence_interval"]
    assert lo <= est <= hi
