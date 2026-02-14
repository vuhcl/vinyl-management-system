"""
Evaluation metrics: Macro-F1 (primary), Accuracy.
"""
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Macro-averaged F1."""
    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=0, **kwargs)
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Accuracy score."""
    return float(accuracy_score(y_true, y_pred, **kwargs))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute macro_f1 and accuracy. Keys prefixed with prefix (e.g. 'sleeve_', 'media_').
    """
    p = prefix + "_" if prefix else ""
    return {
        f"{p}macro_f1": macro_f1(y_true, y_pred),
        f"{p}accuracy": accuracy(y_true, y_pred),
    }
