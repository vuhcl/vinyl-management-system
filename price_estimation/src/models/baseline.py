"""
Baseline: Linear Regression for vinyl price estimation.
"""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class BaselinePriceModel:
    """
    Linear regression model for price prediction.
    Supports save/load and prediction in user-story JSON format.
    """

    MODEL_VERSION = "v1.0"

    def __init__(
        self,
        fit_intercept: bool = True,
        positive: bool = False,
        feature_names: list[str] | None = None,
    ):
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.feature_names_ = feature_names or []
        self.model_: LinearRegression | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "BaselinePriceModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        y = np.asarray(y).ravel()
        self.model_ = LinearRegression(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame):
            X = X.reindex(columns=self.feature_names_, fill_value=0).values
        return self.model_.predict(X).ravel()

    def predict_item(
        self,
        item_id: str,
        X: np.ndarray | pd.DataFrame,
        prediction_interval: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """
        Return single-item prediction in user-story JSON format.
        prediction_interval can be (low, high) from residual-based interval if available.
        """
        pred = self.predict(X)
        price = float(pred[0]) if pred.size else 0.0
        if prediction_interval is not None:
            low, high = prediction_interval
            interval = [round(low, 2), round(high, 2)]
        else:
            interval = [round(price * 0.9, 2), round(price * 1.1, 2)]
        return {
            "item_id": str(item_id),
            "predicted_price": round(price, 2),
            "prediction_interval": interval,
            "model_version": self.MODEL_VERSION,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path / "model.joblib")
        joblib.dump(self.feature_names_, path / "feature_names.joblib")
        joblib.dump(
            {"fit_intercept": self.fit_intercept, "positive": self.positive},
            path / "config.joblib",
        )

    @classmethod
    def load(cls, path: Path | str) -> "BaselinePriceModel":
        path = Path(path)
        self = cls()
        self.model_ = joblib.load(path / "model.joblib")
        self.feature_names_ = joblib.load(path / "feature_names.joblib")
        cfg = joblib.load(path / "config.joblib")
        self.fit_intercept = cfg.get("fit_intercept", True)
        self.positive = cfg.get("positive", False)
        return self


def train_baseline(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    fit_intercept: bool = True,
    positive: bool = False,
) -> BaselinePriceModel:
    """Train and return a BaselinePriceModel (Linear Regression)."""
    model = BaselinePriceModel(fit_intercept=fit_intercept, positive=positive)
    return model.fit(X, y)
