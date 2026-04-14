"""MLflow pyfunc model: log1p regressor + condition adjustment for batch scoring."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from ..features.vinyliq_features import (
    condition_string_to_ordinal,
    default_feature_columns,
    grade_delta_scale_params_from_cond,
    residual_training_feature_columns,
    scaled_condition_log_adjustment,
)
from .condition_adjustment import default_params, load_params_with_grade_delta_overlays
from .fitted_regressor import TARGET_KIND_RESIDUAL_LOG_MEDIAN

# Residual reconstruction anchor: prefer release lowest, same order as training.
_PYFUNC_MEDIAN_COL = "discogs_median_price"


class VinylIQPricePyFunc(PythonModel):
    """Loaded via mlflow.pyfunc.load_model or mlflow models serve."""

    def load_context(self, context: Any) -> None:
        import joblib

        art = context.artifacts
        manifest_path = Path(art["manifest"])
        reg_path = Path(art["regressor"])
        feat_path = Path(art["feature_columns"])
        target_path = Path(art["target_log1p"])
        cond_path = Path(art["condition_params"])

        manifest = json.loads(manifest_path.read_text())
        self._backend = str(manifest.get("backend", "unknown"))
        schema = int(manifest.get("schema_version", 1))
        tk = str(manifest.get("target_kind", "")).strip()
        if schema >= 2 and tk:
            self._target_kind = tk
        else:
            self._target_kind = "dollar_log1p"
        self._estimator = joblib.load(reg_path)
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
        df = model_input
        cols = self._feature_columns
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:10]!r}…")

        X = df[cols].to_numpy(dtype=np.float64, copy=False)
        logp = np.asarray(self._estimator.predict(X), dtype=np.float64).ravel()
        n = len(df)
        if self._target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            if _PYFUNC_MEDIAN_COL not in df.columns:
                raise ValueError(
                    f"Residual target requires {_PYFUNC_MEDIAN_COL!r} column "
                    f"(log1p anchor; use release or marketplace lowest)"
                )
            med = np.maximum(
                df[_PYFUNC_MEDIAN_COL].to_numpy(dtype=np.float64, copy=False),
                0.0,
            )
            logp = logp + np.log1p(med)
            anchor_arr = np.maximum(med, 1e-6)
        elif _PYFUNC_MEDIAN_COL in df.columns:
            anchor_arr = np.maximum(
                df[_PYFUNC_MEDIAN_COL].to_numpy(dtype=np.float64, copy=False),
                1e-6,
            )
        else:
            anchor_arr = np.ones(n, dtype=np.float64)
        if "media_condition" in df.columns:
            media_ord = np.array(
                [condition_string_to_ordinal(x) for x in df["media_condition"]],
                dtype=np.float64,
            )
        else:
            media_ord = df["media_grade"].to_numpy(dtype=np.float64, copy=False)
        if "sleeve_condition" in df.columns:
            sleeve_ord = np.array(
                [condition_string_to_ordinal(x) for x in df["sleeve_condition"]],
                dtype=np.float64,
            )
        else:
            sleeve_ord = df["sleeve_grade"].to_numpy(dtype=np.float64, copy=False)

        alpha = float(self._cond.get("alpha", -0.06))
        beta = float(self._cond.get("beta", -0.04))
        ref_grade = float(self._cond.get("ref_grade", 8.0))
        scale_params = grade_delta_scale_params_from_cond(self._cond)
        has_year = "year" in df.columns
        year_col = df["year"].to_numpy(dtype=np.float64, copy=False) if has_year else None

        prices = []
        for i in range(n):
            anchor_i = float(anchor_arr[i])
            if anchor_i <= 0.0:
                anchor_i = 1.0
            yr = float(year_col[i]) if year_col is not None else None
            if yr is not None and not np.isfinite(yr):
                yr = None
            logp_adj = scaled_condition_log_adjustment(
                float(logp[i]),
                float(media_ord[i]),
                float(sleeve_ord[i]),
                base_alpha=alpha,
                base_beta=beta,
                ref_grade=ref_grade,
                anchor_usd=anchor_i,
                release_year=yr,
                scale_params=scale_params,
            )
            prices.append(float(np.expm1(np.clip(logp_adj, 0, 25))))

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
    return {
        "manifest": str(d / "model_manifest.json"),
        "regressor": str(d / "regressor.joblib"),
        "feature_columns": str(d / "feature_columns.joblib"),
        "target_log1p": str(d / "target_log1p.joblib"),
        "condition_params": str(d / "condition_params.json"),
    }


def build_pyfunc_input_example(
    *,
    target_kind: str | None = None,
) -> pd.DataFrame:
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        cols = residual_training_feature_columns()
    else:
        cols = default_feature_columns()
    row = {c: 0.0 for c in cols}
    row["media_condition"] = "Near Mint (NM or M-)"
    row["sleeve_condition"] = "Near Mint (NM or M-)"
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        row[_PYFUNC_MEDIAN_COL] = 25.0
    row["year"] = 2000.0
    return pd.DataFrame([row])
