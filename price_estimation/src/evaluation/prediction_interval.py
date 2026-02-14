"""
Prediction interval evaluation: coverage and interval width.
"""
import numpy as np


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Fraction of true values that fall within [lower, upper].
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean interval width (upper - lower)."""
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    return float(np.mean(upper - lower))
