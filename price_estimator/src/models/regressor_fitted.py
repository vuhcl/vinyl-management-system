"""Fitted VinylIQ regressor wrapper shared by training and inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..features.vinyliq_features import MAX_LOG_PRICE
from .regressor_constants import (
    FEATURE_COLUMNS_FILE,
    MANIFEST_FILE,
    REGRESSOR_FILE,
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    TARGET_LOG1P_FILE,
    LEGACY_XGB_FILE,
)


@dataclass
class FittedVinylIQRegressor:
    """Thin wrapper so inference and pyfunc share one predict path."""

    backend: str
    estimator: Any
    feature_columns: list[str]
    target_was_log1p: bool = True
    target_kind: str = TARGET_KIND_DOLLAR_LOG1P

    def predict_log1p(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("Model not fitted")
        out = self.estimator.predict(X)
        return np.asarray(out, dtype=np.float64).ravel()

    def predict_dollars(self, X: np.ndarray) -> np.ndarray:
        logp = self.predict_log1p(X)
        if self.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            raise TypeError(
                "predict_dollars requires median anchor; use inference service or "
                "log1p_dollar_from_residual(predict_log1p(X), median) then expm1"
            )
        if self.target_was_log1p:
            return np.expm1(np.clip(logp, 0, MAX_LOG_PRICE))
        return np.clip(logp, 0, None)

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 2,
            "backend": self.backend,
            "target_kind": self.target_kind,
        }
        (d / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        joblib.dump(self.estimator, d / REGRESSOR_FILE)
        joblib.dump(self.feature_columns, d / FEATURE_COLUMNS_FILE)
        joblib.dump(
            self.target_kind == TARGET_KIND_DOLLAR_LOG1P and self.target_was_log1p,
            d / TARGET_LOG1P_FILE,
        )
        if self.backend == "xgboost":
            joblib.dump(self.estimator, d / LEGACY_XGB_FILE)
