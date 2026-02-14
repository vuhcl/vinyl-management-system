"""
Advanced: Gradient Boosting (LightGBM) for price estimation with
prediction intervals.
"""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class GradientBoostingPriceModel:
    """
    LightGBM regressor with optional quantile prediction for intervals.
    """

    MODEL_VERSION = "v1.0"

    def __init__(
        self,
        quantiles: list[float] | None = None,
        feature_names: list[str] | None = None,
    ):
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.feature_names_ = feature_names or []
        self.model_median_: Any = None
        self.models_quantile_: dict[float, Any] = {}

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        random_state: int = 42,
    ) -> "GradientBoostingPriceModel":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "lightgbm required for gradient_boosting. pip install lightgbm"
            )
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        y = np.asarray(y).ravel()
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "verbosity": -1,
        }
        self.model_median_ = lgb.LGBMRegressor(objective="regression", **params)
        self.model_median_.fit(X, y)
        for q in self.quantiles:
            if 0 < q < 1 and q != 0.5:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
                m.fit(X, y)
                self.models_quantile_[q] = m
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model_median_ is None:
            raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame):
            X = X.reindex(
                columns=self.feature_names_, fill_value=0
            ).values
        return self.model_median_.predict(X).ravel()

    def predict_interval(
        self,
        X: np.ndarray | pd.DataFrame,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point_pred, lower, upper) for prediction intervals."""
        point = self.predict(X)
        if isinstance(X, pd.DataFrame):
            X = X.reindex(columns=self.feature_names_, fill_value=0).values
        lower = point.copy()
        upper = point.copy()
        if lower_quantile in self.models_quantile_:
            lower = self.models_quantile_[lower_quantile].predict(X).ravel()
        if upper_quantile in self.models_quantile_:
            upper = self.models_quantile_[upper_quantile].predict(X).ravel()
        return point, lower, upper

    def predict_item(
        self,
        item_id: str,
        X: np.ndarray | pd.DataFrame,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9,
    ) -> dict[str, Any]:
        point, low, high = self.predict_interval(X, lower_quantile, upper_quantile)
        price = float(point[0]) if point.size else 0.0
        return {
            "item_id": str(item_id),
            "predicted_price": round(price, 2),
            "prediction_interval": [
                round(float(low[0]), 2), round(float(high[0]), 2)
            ],
            "model_version": self.MODEL_VERSION,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_median_, path / "model_median.joblib")
        joblib.dump(self.models_quantile_, path / "models_quantile.joblib")
        joblib.dump(self.feature_names_, path / "feature_names.joblib")
        joblib.dump(self.quantiles, path / "quantiles.joblib")

    @classmethod
    def load(cls, path: Path | str) -> "GradientBoostingPriceModel":
        path = Path(path)
        self = cls()
        self.model_median_ = joblib.load(path / "model_median.joblib")
        self.models_quantile_ = joblib.load(path / "models_quantile.joblib")
        self.feature_names_ = joblib.load(path / "feature_names.joblib")
        self.quantiles = joblib.load(path / "quantiles.joblib")
        return self


def train_gradient_boosting(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_samples: int = 20,
    prediction_interval_quantiles: list[float] | None = None,
    random_state: int = 42,
) -> GradientBoostingPriceModel:
    """Train GradientBoostingPriceModel (LightGBM) with optional quantiles."""
    quantiles = prediction_interval_quantiles or [0.1, 0.5, 0.9]
    model = GradientBoostingPriceModel(quantiles=quantiles)
    return model.fit(
        X, y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        random_state=random_state,
    )
