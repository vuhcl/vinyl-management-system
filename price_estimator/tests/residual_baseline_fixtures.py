"""Generate `residual_reconstruction_baseline.json` for R0 / R8 regression lock."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from price_estimator.src.features.vinyliq_features import residual_training_feature_columns
from price_estimator.src.models.condition_adjustment import default_params, save_params
from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
    pred_log1p_dollar_for_metrics,
)
from price_estimator.src.models.vinyliq_pyfunc import VinylIQPricePyFunc, pyfunc_artifacts_dict

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "residual_reconstruction_baseline.json"


def _repr_float(x: float) -> str:
    return repr(float(np.float64(x)))


def _build_residual_pyfunc_df(meds: list[float]) -> pd.DataFrame:
    cols = residual_training_feature_columns()
    rows = []
    for med in meds:
        row = {c: 0.0 for c in cols}
        row["media_grade"] = 7.0
        row["sleeve_grade"] = 7.0
        row["condition_discount"] = 7.0 / 8.0
        row["media_condition"] = "Near Mint (NM or M-)"
        row["sleeve_condition"] = "Near Mint (NM or M-)"
        row["discogs_median_price"] = float(med)
        row["year"] = 2000.0
        rows.append(row)
    return pd.DataFrame(rows)


def _pred_log1p_grid_baseline() -> list[dict[str, str | float]]:
    preds = [-2.0, -0.5, 0.0, 0.5, 2.0]
    meds = [0.0, 1.0, 5.0, 25.0, 100.0, 1000.0]
    kinds = [TARGET_KIND_DOLLAR_LOG1P, TARGET_KIND_RESIDUAL_LOG_MEDIAN]
    out: list[dict[str, str | float]] = []
    for kind in kinds:
        for p in preds:
            for m in meds:
                lp = pred_log1p_dollar_for_metrics(
                    np.array([p], dtype=np.float64),
                    np.array([m], dtype=np.float64),
                    kind,
                )
                out.append(
                    {
                        "pred": p,
                        "med": m,
                        "kind": kind,
                        "out": _repr_float(float(lp[0])),
                    }
                )
    return out


def _pyfunc_prices(*, ensemble: dict | None, meds: list[float]) -> list[str]:
    tmp_path = Path(tempfile.mkdtemp(prefix="r0_pyfunc_"))
    try:
        cols = residual_training_feature_columns()
        n = len(meds)
        X = np.zeros((n, len(cols)))

        if ensemble is None:
            dummy = DummyRegressor(strategy="constant", constant=0.0)
            dummy.fit(X, np.zeros(n))
            reg = FittedVinylIQRegressor(
                "sklearn_rf",
                dummy,
                cols,
                target_was_log1p=False,
                target_kind=TARGET_KIND_RESIDUAL_LOG_MEDIAN,
            )
            reg.save(tmp_path)
        else:
            d_nm = DummyRegressor(strategy="constant", constant=0.2)
            d_ord = DummyRegressor(strategy="constant", constant=-0.2)
            d_nm.fit(X, np.full(n, 0.2))
            d_ord.fit(X, np.full(n, -0.2))
            joblib.dump(d_nm, tmp_path / "regressor_ensemble_nm.joblib")
            joblib.dump(d_ord, tmp_path / "regressor_ensemble_ord.joblib")
            joblib.dump(d_ord, tmp_path / "regressor.joblib")
            joblib.dump(cols, tmp_path / "feature_columns.joblib")
            joblib.dump(False, tmp_path / "target_log1p.joblib")
            manifest = {
                "schema_version": 3,
                "backend": "sklearn_rf",
                "target_kind": TARGET_KIND_RESIDUAL_LOG_MEDIAN,
                "ensemble": ensemble,
            }
            (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))

        save_params(tmp_path / "condition_params.json", default_params())
        arts = pyfunc_artifacts_dict(tmp_path)
        model = VinylIQPricePyFunc()
        model.load_context(type("Ctx", (), {"artifacts": arts})())
        df = _build_residual_pyfunc_df(meds)
        out_df = model.predict(object(), df)
        return [_repr_float(float(out_df["estimated_price"].iloc[i])) for i in range(n)]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def generate_baseline_dict() -> dict:
    meds6 = [0.0, 1.0, 5.0, 25.0, 100.0, 1000.0]
    return {
        "version": 1,
        "pred_log1p_dollar_for_metrics": _pred_log1p_grid_baseline(),
        "pyfunc_single_residual_6row": _pyfunc_prices(ensemble=None, meds=meds6),
        "pyfunc_ensemble_residual_t4_s1": _pyfunc_prices(
            ensemble={
                "enabled": True,
                "blend": {"kind": "log_anchor_sigmoid", "t": 4.0, "s": 1.0},
                "regressor_nm": "regressor_ensemble_nm.joblib",
                "regressor_ord": "regressor_ensemble_ord.joblib",
            },
            meds=meds6,
        ),
        "pyfunc_ensemble_residual_t2.5_s0.5": _pyfunc_prices(
            ensemble={
                "enabled": True,
                "blend": {"kind": "log_anchor_sigmoid", "t": 2.5, "s": 0.5},
                "regressor_nm": "regressor_ensemble_nm.joblib",
                "regressor_ord": "regressor_ensemble_ord.joblib",
            },
            meds=meds6,
        ),
    }


def write_baseline_fixture() -> Path:
    data = generate_baseline_dict()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return FIXTURE_PATH
