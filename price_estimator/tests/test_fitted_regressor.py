"""Round-trip save/load for FittedVinylIQRegressor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from price_estimator.src.features.vinyliq_features import residual_training_feature_columns
from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
    apply_format_multipliers_to_weights,
    combine_anchor_and_format_sample_weights,
    ensemble_blend_weight_log_anchor,
    fit_regressor,
    load_fitted_regressor,
    mae_dollars,
    median_ape_dollars,
    median_ape_dollar_quartiles,
    median_ape_quartile_format_slice_diagnostics,
    median_ape_quartile_format_slice_table,
    median_ape_train_median_baseline,
    metrics_dollar_from_log1p_masked,
    mutually_exclusive_format_bucket_masks,
    true_dollar_quartile_masks,
    training_sample_weights_from_anchors,
    wape_dollars,
    weighted_format_median_ape_dollars,
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


def test_true_dollar_quartile_masks_partition_rows() -> None:
    rng = np.random.default_rng(7)
    yt = rng.uniform(10.0, 200.0, size=200)
    masks = true_dollar_quartile_masks(yt, n_bins=4)
    assert len(masks) == 4
    union = np.zeros_like(yt, dtype=bool)
    for m in masks:
        assert not np.any(union & m)
        union |= m
    assert np.all(union)


def test_format_bucket_masks_partition_rows() -> None:
    cols = residual_training_feature_columns()
    n = 100
    X = np.zeros((n, len(cols)))
    rng = np.random.default_rng(8)
    X[:, cols.index("is_lp")] = rng.integers(0, 2, size=n).astype(np.float64)
    X[:, cols.index("is_7inch")] = rng.integers(0, 2, size=n).astype(np.float64)
    b = mutually_exclusive_format_bucket_masks(X, cols)
    union = np.zeros(n, dtype=bool)
    for name, m in b.items():
        assert not np.any(union & m), name
        union |= m
    assert np.all(union)


def test_slice_diagnostics_median_matches_slice_table() -> None:
    cols = residual_training_feature_columns()
    n = 120
    rng = np.random.default_rng(9)
    y = np.log1p(rng.uniform(5.0, 120.0, size=n))
    pred = y + 0.02 * rng.standard_normal(n)
    X = np.zeros((n, len(cols)))
    X[:, cols.index("is_lp")] = 1.0
    tab = median_ape_quartile_format_slice_table(y, pred, X, cols, min_count=5)
    diag = median_ape_quartile_format_slice_diagnostics(y, pred, X, cols, min_count=5)
    by_key = {(d["quartile"], d["slice"]): d for d in diag}
    for r in tab:
        if r["n_rows"] < 5 or np.isnan(r["median_ape"]):
            continue
        d = by_key[(r["quartile"], r["slice"])]
        assert d["median_ape"] == pytest.approx(float(r["median_ape"]))


def test_one_decimal_percent_hides_small_median_ape() -> None:
    """Training prints ``100*md:.1f``; sub-0.05% medians show as 0.0% without a bug."""
    md = 0.003 / 100.0
    assert f"{100.0 * md:.1f}" == "0.0"


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


def test_ensemble_blend_weight_increases_with_anchor() -> None:
    cheap = np.array([5.0, 10.0])
    pricey = np.array([80.0, 200.0])
    w_lo = ensemble_blend_weight_log_anchor(
        cheap, center_log1p=4.0, scale=0.5
    )
    w_hi = ensemble_blend_weight_log_anchor(
        pricey, center_log1p=4.0, scale=0.5
    )
    assert float(np.mean(w_hi)) > float(np.mean(w_lo))
    assert bool(np.all(w_lo >= 0.0) and np.all(w_lo <= 1.0))


def test_metrics_dollar_from_log1p_masked_respects_min_count() -> None:
    y = np.log1p(np.array([20.0, 25.0, 30.0], dtype=np.float64))
    p = y + 0.01
    mask = np.array([True, False, False])
    mae, wape, md = metrics_dollar_from_log1p_masked(
        y, p, mask, min_count=5
    )
    assert np.isnan(mae) and np.isnan(wape) and np.isnan(md)


def test_apply_format_multipliers_increases_lp_weight() -> None:
    cols = residual_training_feature_columns()
    n = 10
    X = np.zeros((n, len(cols)))
    X[:5, cols.index("is_lp")] = 1.0
    X[5:, cols.index("is_cd")] = 1.0
    base = np.ones(n)
    w = apply_format_multipliers_to_weights(
        base,
        X,
        cols,
        {"default": 1.0, "lp": 2.0},
    )
    assert w is not None
    assert np.mean(w) == pytest.approx(1.0)
    lp_m = X[:, cols.index("is_lp")] >= 0.5
    cd_m = X[:, cols.index("is_cd")] >= 0.5
    assert float(np.mean(w[lp_m])) > float(np.mean(w[cd_m]))


def test_combine_anchor_format_weights_mean_one() -> None:
    cols = residual_training_feature_columns()
    n = 8
    X = np.zeros((n, len(cols)))
    X[:, cols.index("is_cd")] = 1.0
    med = np.array([16.0, 25.0, 100.0, 4.0, 9.0, 36.0, 49.0, 4.0])
    w = combine_anchor_and_format_sample_weights(
        med,
        "inv_sqrt_anchor",
        X,
        cols,
        {"default": 1.0, "cd": 1.5},
    )
    assert w is not None
    assert np.mean(w) == pytest.approx(1.0)


def test_weighted_format_median_ape_finite_per_bucket() -> None:
    cols = residual_training_feature_columns()
    n = 80
    rng = np.random.default_rng(42)
    y_lp = np.log1p(rng.uniform(10.0, 80.0, size=n))
    pred_lp = y_lp.copy()
    pred_lp[:40] += 0.08
    X = np.zeros((n, len(cols)))
    X[:40, cols.index("is_lp")] = 1.0
    X[40:, cols.index("is_7inch")] = 1.0
    wf = weighted_format_median_ape_dollars(
        y_lp,
        pred_lp,
        X,
        cols,
        {"default": 1.0, "lp": 4.0, "seven": 1.0},
        min_count=15,
    )
    assert np.isfinite(wf)


def test_slice_percent_one_decimal_can_hide_small_mdape() -> None:
    """Ablation tip: compare slice diagnostics (p90/max) when cells show 0.0%."""
    md = 0.003 / 100.0
    assert f"{100.0 * md:.1f}" == "0.0"
