"""MLflow pyfunc model: log1p regressor + condition adjustment for batch scoring."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from ..features.vinyliq_features import (
    MAX_LOG_PRICE,
    clamp_ordinals_for_inference,
    condition_string_to_ordinal,
    default_feature_columns,
    grade_delta_scale_params_from_cond,
    residual_training_feature_columns,
    scaled_condition_log_adjustment,
)
from .condition_adjustment import default_params, load_params_with_grade_delta_overlays
from .fitted_regressor import TARGET_KIND_RESIDUAL_LOG_MEDIAN, ensemble_blend_weight_log_anchor
from .model_manifest import ModelManifest
from .residual_dollar_reconstruction import pred_log1p_dollar_from_stored_prediction

# Residual reconstruction anchor: prefer release lowest, same order as training.
_PYFUNC_MEDIAN_COL = "discogs_median_price"
_PYFUNC_MIN_PRICE_USD = 0.50
_PYFUNC_MIN_RELEASE_YEAR = 1877
_PYFUNC_MAX_RELEASE_YEAR = 2030


class VinylIQPricePyFunc(PythonModel):
    """Loaded via mlflow.pyfunc.load_model or mlflow models serve."""

    def load_context(self, context: Any) -> None:
        import joblib

        art = context.artifacts
        manifest_path = Path(art["manifest"])
        feat_path = Path(art["feature_columns"])
        target_path = Path(art["target_log1p"])
        cond_path = Path(art["condition_params"])

        manifest = json.loads(manifest_path.read_text())
        mm = ModelManifest.from_dict(manifest)
        self._backend = mm.backend or "unknown"
        self._target_kind = mm.target_kind
        self._ensemble = mm.ensemble
        self._estimator = None
        self._estimator_nm = None
        self._estimator_ord = None
        self._blend_t = 4.0
        self._blend_s = 0.35
        if self._ensemble is not None:
            blend = self._ensemble.get("blend") or {}
            self._blend_t = float(blend.get("t", 4.0))
            self._blend_s = float(blend.get("s", 0.35))
            self._estimator_nm = joblib.load(
                Path(art["regressor_ensemble_nm"]),
            )
            self._estimator_ord = joblib.load(
                Path(art["regressor_ensemble_ord"]),
            )
        else:
            self._estimator = joblib.load(Path(art["regressor"]))

        self._feature_columns: list[str] = list(joblib.load(feat_path))
        self._target_log1p = bool(joblib.load(target_path))
        self._cond = (
            load_params_with_grade_delta_overlays(Path(cond_path).parent)
            if cond_path.is_file()
            else default_params()
        )

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        _ = params
        df_work = model_input.copy()
        cols = self._feature_columns
        missing = [c for c in cols if c not in df_work.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:10]!r}…")

        if "media_condition" in df_work.columns:
            media_ord = np.array(
                [condition_string_to_ordinal(x) for x in df_work["media_condition"]],
                dtype=np.float64,
            )
        else:
            media_ord = df_work["media_grade"].to_numpy(dtype=np.float64, copy=True)
        if "sleeve_condition" in df_work.columns:
            sleeve_ord = np.array(
                [condition_string_to_ordinal(x) for x in df_work["sleeve_condition"]],
                dtype=np.float64,
            )
        else:
            sleeve_ord = df_work["sleeve_grade"].to_numpy(dtype=np.float64, copy=True)

        nm_grade = 7.0
        nm_discount = nm_grade / 8.0
        if "media_grade" in df_work.columns:
            df_work["media_grade"] = nm_grade
        if "sleeve_grade" in df_work.columns:
            df_work["sleeve_grade"] = nm_grade
        if "condition_discount" in df_work.columns:
            df_work["condition_discount"] = nm_discount

        X = df_work[cols].to_numpy(dtype=np.float64, copy=False)
        n = len(df_work)
        if self._ensemble is not None:
            if self._estimator_nm is None or self._estimator_ord is None:
                raise RuntimeError("Ensemble manifest missing estimator artifacts")
            z_nm = np.asarray(
                self._estimator_nm.predict(X), dtype=np.float64
            ).ravel()
            z_ord = np.asarray(
                self._estimator_ord.predict(X), dtype=np.float64
            ).ravel()
            if self._target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
                if _PYFUNC_MEDIAN_COL not in model_input.columns:
                    raise ValueError(
                        f"Residual target requires {_PYFUNC_MEDIAN_COL!r} column "
                        f"(log1p anchor; use release or marketplace lowest)"
                    )
                med = np.maximum(
                    model_input[_PYFUNC_MEDIAN_COL].to_numpy(
                        dtype=np.float64, copy=False
                    ),
                    0.0,
                )
                lp_nm = pred_log1p_dollar_from_stored_prediction(
                    z_nm, med, self._target_kind
                )
                lp_ord = pred_log1p_dollar_from_stored_prediction(
                    z_ord, med, self._target_kind
                )
                w = ensemble_blend_weight_log_anchor(
                    med, center_log1p=self._blend_t, scale=self._blend_s
                )
                logp = w * lp_nm + (1.0 - w) * lp_ord
            else:
                raise ValueError(
                    "vinyliq ensemble is only supported for residual_log_median targets"
                )
            anchor_arr = np.maximum(
                model_input[_PYFUNC_MEDIAN_COL].to_numpy(
                    dtype=np.float64, copy=False
                ),
                1e-6,
            ) if _PYFUNC_MEDIAN_COL in model_input.columns else np.ones(
                n, dtype=np.float64
            )
        else:
            assert self._estimator is not None
            logp = np.asarray(self._estimator.predict(X), dtype=np.float64).ravel()
            if self._target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
                if _PYFUNC_MEDIAN_COL not in model_input.columns:
                    raise ValueError(
                        f"Residual target requires {_PYFUNC_MEDIAN_COL!r} column "
                        f"(log1p anchor; use release or marketplace lowest)"
                    )
                med = np.maximum(
                    model_input[_PYFUNC_MEDIAN_COL].to_numpy(
                        dtype=np.float64, copy=False
                    ),
                    0.0,
                )
                logp = pred_log1p_dollar_from_stored_prediction(
                    logp, med, self._target_kind
                )
                anchor_arr = np.maximum(med, 1e-6)
            elif _PYFUNC_MEDIAN_COL in model_input.columns:
                anchor_arr = np.maximum(
                    model_input[_PYFUNC_MEDIAN_COL].to_numpy(
                        dtype=np.float64, copy=False
                    ),
                    1e-6,
                )
            else:
                anchor_arr = np.ones(n, dtype=np.float64)
        _dp = default_params()
        alpha = float(self._cond.get("alpha", _dp["alpha"]))
        beta = float(self._cond.get("beta", _dp["beta"]))
        ref_grade = float(self._cond.get("ref_grade", 8.0))
        scale_params = grade_delta_scale_params_from_cond(self._cond)
        has_year = "year" in model_input.columns
        year_col = (
            model_input["year"].to_numpy(dtype=np.float64, copy=False)
            if has_year
            else None
        )

        prices = []
        for i in range(n):
            anchor_i = float(anchor_arr[i])
            if anchor_i <= 0.0:
                anchor_i = 1.0
            yr = float(year_col[i]) if year_col is not None else None
            if yr is not None and not np.isfinite(yr):
                yr = None
            elif yr is not None:
                yr = float(
                    max(
                        _PYFUNC_MIN_RELEASE_YEAR,
                        min(_PYFUNC_MAX_RELEASE_YEAR, yr),
                    )
                )
            mo, so = clamp_ordinals_for_inference(
                float(media_ord[i]),
                float(sleeve_ord[i]),
            )
            logp_adj = scaled_condition_log_adjustment(
                float(logp[i]),
                mo,
                so,
                base_alpha=alpha,
                base_beta=beta,
                ref_grade=ref_grade,
                anchor_usd=anchor_i,
                release_year=yr,
                scale_params=scale_params,
            )
            raw = float(np.expm1(np.clip(logp_adj, 0, MAX_LOG_PRICE)))
            prices.append(max(raw, _PYFUNC_MIN_PRICE_USD))

        out = pd.DataFrame(
            {
                "estimated_price": prices,
                "model_backend": [self._backend] * n,
            }
        )
        return out


def pyfunc_artifacts_dict(model_dir: Path) -> dict[str, str]:
    """Artifact paths for mlflow.pyfunc.log_model (all must exist)."""
    d = Path(model_dir)
    out: dict[str, str] = {
        "manifest": str(d / "model_manifest.json"),
        "regressor": str(d / "regressor.joblib"),
        "feature_columns": str(d / "feature_columns.joblib"),
        "target_log1p": str(d / "target_log1p.joblib"),
        "condition_params": str(d / "condition_params.json"),
    }
    mf = d / "model_manifest.json"
    if mf.is_file():
        try:
            man = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            return out
        ens = man.get("ensemble")
        if isinstance(ens, dict) and ens.get("enabled"):
            nm = str(ens.get("regressor_nm", "regressor_ensemble_nm.joblib"))
            od = str(ens.get("regressor_ord", "regressor_ensemble_ord.joblib"))
            out["regressor_ensemble_nm"] = str(d / nm)
            out["regressor_ensemble_ord"] = str(d / od)
    return out


def build_pyfunc_input_example(
    *,
    target_kind: str | None = None,
) -> pd.DataFrame:
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        cols = residual_training_feature_columns()
    else:
        cols = default_feature_columns()
    row = {c: 0.0 for c in cols}
    row["media_grade"] = 7.0
    row["sleeve_grade"] = 7.0
    row["condition_discount"] = 7.0 / 8.0
    row["media_condition"] = "Near Mint (NM or M-)"
    row["sleeve_condition"] = "Near Mint (NM or M-)"
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        row[_PYFUNC_MEDIAN_COL] = 25.0
    row["year"] = 2000.0
    return pd.DataFrame([row])
