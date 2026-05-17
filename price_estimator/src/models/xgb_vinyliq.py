"""XGBoost regressor for VinylIQ (log1p price target)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..features.vinyliq_features import MAX_LOG_PRICE


class XGBVinylIQModel:
    def __init__(self) -> None:
        self.model_: Any = None
        self.feature_columns_: list[str] = []
        self.target_was_log1p_: bool = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_columns: list[str],
        xgb_params: dict[str, Any] | None = None,
    ) -> XGBVinylIQModel:
        import xgboost as xgb

        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "verbosity": 0,
        }
        if xgb_params:
            params.update(xgb_params)
        self.feature_columns_ = list(feature_columns)
        self.model_ = xgb.XGBRegressor(**params)
        self.model_.fit(X, y)
        return self

    def predict_log1p(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        return self.model_.predict(X)

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, d / "xgb_model.joblib")
        joblib.dump(self.feature_columns_, d / "feature_columns.joblib")
        joblib.dump(self.target_was_log1p_, d / "target_log1p.joblib")

    @classmethod
    def load(cls, directory: Path | str) -> XGBVinylIQModel:
        d = Path(directory)
        self = cls()
        self.model_ = joblib.load(d / "xgb_model.joblib")
        self.feature_columns_ = joblib.load(d / "feature_columns.joblib")
        self.target_was_log1p_ = joblib.load(d / "target_log1p.joblib")
        return self

    def predict_dollars(self, X: np.ndarray) -> np.ndarray:
        logp = self.predict_log1p(X)
        if self.target_was_log1p_:
            return np.expm1(np.clip(logp, 0, MAX_LOG_PRICE))
        return np.clip(logp, 0, None)
