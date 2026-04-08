"""Round-trip save/load for FittedVinylIQRegressor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
    fit_regressor,
    load_fitted_regressor,
    mae_dollars,
    median_ape_dollars,
    median_ape_dollar_quartiles,
    median_ape_train_median_baseline,
    wape_dollars,
)


def test_xgboost_fit_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5))
    y = np.log1p(rng.random(40) * 50)
    cols = [f"f{i}" for i in range(5)]
    reg, _ = fit_regressor(
        "xgboost",
        {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.2},
        X,
        y,
        cols,
        random_state=0,
    )
    reg.save(tmp_path)
    loaded = load_fitted_regressor(tmp_path)
    assert loaded is not None
    assert loaded.backend == "xgboost"
    np.testing.assert_allclose(reg.predict_log1p(X), loaded.predict_log1p(X), rtol=1e-5)


def test_residual_target_fit_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 5))
    med = rng.random(40) * 30 + 5
    y_dollar = np.log1p(rng.random(40) * 50)
    y_resid = y_dollar - np.log1p(med)
    cols = [f"f{i}" for i in range(5)]
    reg, _ = fit_regressor(
        "xgboost",
        {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.2},
        X,
        y_resid,
        cols,
        random_state=0,
        target_kind=TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    )
    reg.save(tmp_path)
    loaded = load_fitted_regressor(tmp_path)
    assert loaded is not None
    assert loaded.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN
    np.testing.assert_allclose(reg.predict_log1p(X), loaded.predict_log1p(X), rtol=1e-5)


def test_mae_dollars() -> None:
    y_true = np.log1p(np.array([10.0, 20.0]))
    pred = np.log1p(np.array([12.0, 18.0]))
    m = mae_dollars(y_true, pred)
    assert m == pytest.approx(2.0)


def test_wape_dollars() -> None:
    y_true = np.log1p(np.array([10.0, 20.0]))
    pred = np.log1p(np.array([12.0, 18.0]))
    # |e|: 2,2 sum=4; sum|y|=30
    assert wape_dollars(y_true, pred) == pytest.approx(4.0 / 30.0)


def test_median_ape_dollars() -> None:
    y_true = np.log1p(np.array([10.0, 20.0]))
    pred = np.log1p(np.array([12.0, 18.0]))
    # APE: 0.2, 0.1 → median 0.15
    assert median_ape_dollars(y_true, pred) == pytest.approx(0.15)


def test_median_ape_train_median_baseline_perfect() -> None:
    y_tr = np.log1p(np.array([10.0, 10.0, 30.0]))
    y_ev = np.log1p(np.array([10.0, 10.0]))
    # median log1p is log1p(10); predictions exact for 10,10
    assert median_ape_train_median_baseline(y_tr, y_ev) == pytest.approx(0.0)


def test_median_ape_dollar_quartiles_length() -> None:
    rng = np.random.default_rng(0)
    y = np.log1p(rng.uniform(5, 80, size=40))
    pred = y + 0.05 * rng.standard_normal(40)
    qs = median_ape_dollar_quartiles(y, pred, n_bins=4)
    assert len(qs) == 4
    assert all((q >= 0 or np.isnan(q)) for q in qs)
