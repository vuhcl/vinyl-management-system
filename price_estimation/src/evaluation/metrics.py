"""
Evaluation metrics: MAE (primary), MAPE.
"""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. Clips denominator to avoid div by zero."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = np.abs(y_true)
    denom = np.where(denom < epsilon, epsilon, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE and MAPE."""
    return {
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
