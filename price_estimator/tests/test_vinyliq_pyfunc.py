"""Smoke test for MLflow pyfunc model (no MLflow server)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from sklearn.dummy import DummyRegressor

from price_estimator.src.features.vinyliq_features import (
    default_feature_columns,
    residual_training_feature_columns,
)
from price_estimator.src.models.condition_adjustment import default_params, save_params
from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
)
from price_estimator.src.models.vinyliq_pyfunc import (
    VinylIQPricePyFunc,
    build_pyfunc_input_example,
    pyfunc_artifacts_dict,
)


def test_vinyliq_pyfunc_load_and_predict(tmp_path) -> None:
    cols = default_feature_columns()
    X = np.zeros((4, len(cols)))
    y = np.full(4, np.log1p(25.0))
    dummy = DummyRegressor(strategy="constant", constant=float(y[0]))
    dummy.fit(X, y)
    reg = FittedVinylIQRegressor("sklearn_rf", dummy, cols, target_was_log1p=True)
    reg.save(tmp_path)
    save_params(tmp_path / "condition_params.json", default_params())

    ctx = SimpleNamespace(artifacts=pyfunc_artifacts_dict(tmp_path))
    model = VinylIQPricePyFunc()
    model.load_context(ctx)
    sample = build_pyfunc_input_example()
    out = model.predict(ctx, sample)
    assert "estimated_price" in out.columns
    assert len(out) == 1
    assert float(out["estimated_price"].iloc[0]) > 0


def test_vinyliq_pyfunc_residual_adds_median(tmp_path) -> None:
    cols = residual_training_feature_columns()
    X = np.zeros((2, len(cols)))
    dummy = DummyRegressor(strategy="constant", constant=0.0)
    dummy.fit(X, [0.0, 0.0])
    reg = FittedVinylIQRegressor(
        "sklearn_rf",
        dummy,
        cols,
        target_was_log1p=False,
        target_kind=TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    )
    reg.save(tmp_path)
    save_params(tmp_path / "condition_params.json", default_params())

    ctx = SimpleNamespace(artifacts=pyfunc_artifacts_dict(tmp_path))
    model = VinylIQPricePyFunc()
    model.load_context(ctx)
    sample = build_pyfunc_input_example(target_kind=TARGET_KIND_RESIDUAL_LOG_MEDIAN)
    out = model.predict(ctx, sample)
    assert len(out) == 1
    # z=0 + log1p(25) with default condition adjustment → near $25+
    assert float(out["estimated_price"].iloc[0]) > 15.0
