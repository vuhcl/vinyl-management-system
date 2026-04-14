"""Round-trip save/load for FittedVinylIQRegressor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from price_estimator.src.features.vinyliq_features import residual_training_feature_columns
from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
    fit_regressor,
    load_fitted_regressor,
    mae_dollars,
    median_ape_dollars,
    median_ape_dollar_quartiles,
    median_ape_quartile_format_slice_table,
    median_ape_train_median_baseline,
    training_sample_weights_from_anchors,
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


def test_training_sample_weights_inv_sqrt_anchor_mean_one() -> None:
    m = np.array([1.0, 4.0, 100.0])
    w = training_sample_weights_from_anchors(m, "inv_sqrt_anchor")
    assert w is not None
    assert np.mean(w) == pytest.approx(1.0)


def test_training_sample_weights_null() -> None:
    assert training_sample_weights_from_anchors(np.ones(3), None) is None
    assert training_sample_weights_from_anchors(np.ones(3), "off") is None


def test_training_sample_weights_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        training_sample_weights_from_anchors(np.ones(3), "no_such_mode")


def test_median_ape_quartile_format_slice_table_perfect_pred() -> None:
    cols = residual_training_feature_columns()
    n = 40
    rng = np.random.default_rng(1)
    y = np.log1p(rng.uniform(4.0, 90.0, size=n))
    pred = y.copy()
    X = np.zeros((n, len(cols)))
    i_7 = cols.index("is_7inch")
    X[: n // 2, i_7] = 1.0
    rows = median_ape_quartile_format_slice_table(
        y, pred, X, cols, min_count=3
    )
    assert len(rows) == 4 * 7
    for r in rows:
        if r["n_rows"] > 0 and not np.isnan(r["median_ape"]):
            assert r["median_ape"] == pytest.approx(0.0)


def test_fit_regressor_sample_weight_sklearn_rf(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)
    cols = [f"f{i}" for i in range(3)]
    sw = np.ones(30)
    sw[:10] = 3.0
    reg, _ = fit_regressor(
        "sklearn_rf",
        {"n_estimators": 10, "max_depth": 3},
        X,
        y,
        cols,
        random_state=0,
        sample_weight=sw,
    )
    reg.save(tmp_path)
    loaded = load_fitted_regressor(tmp_path)
    assert loaded is not None
