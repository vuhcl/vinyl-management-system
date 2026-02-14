"""
Calibration: reliability diagrams and ECE for probability outputs.
"""
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve as sk_calibration_curve

from ..data.ingest import CONDITION_GRADES


def calibration_curve_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute calibration curve (fraction of positives, mean predicted prob per bin).
    Returns dict with prob_true, prob_pred, bin edges for plotting.
    """
    # For multiclass, use one-vs-rest for the positive class in each bin
    # sklearn.calibration.calibration_curve expects binary (pos_label) or 1d prob of positive class
    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        # Use the predicted class prob (max) and binarize true by predicted class
        pred_class = np.argmax(y_prob, axis=1)
        prob_true, prob_pred = sk_calibration_curve(
            (y_true == np.unique(y_true)[pred_class]).astype(int)
            if len(np.unique(y_true)) == 2
            else (y_true.astype(str) == np.unique(y_true)[pred_class]).astype(int),
            y_prob[np.arange(len(y_prob)), pred_class],
            n_bins=n_bins,
        )
    else:
        prob_true, prob_pred = sk_calibration_curve(
            y_true, y_prob if y_prob.ndim == 1 else y_prob[:, 1], n_bins=n_bins
        )
    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "n_bins": n_bins,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    classes: list[str] | None = None,
) -> float:
    """
    ECE: weighted average of |acc(bin) - conf(bin)|.
    For multiclass, use max prob as confidence and 0/1 correctness.
    y_prob columns must match order of classes (default: CONDITION_GRADES).
    """
    classes = classes or CONDITION_GRADES
    if y_prob.ndim == 2:
        conf = np.max(y_prob, axis=1)
        pred_idx = np.argmax(y_prob, axis=1)
        # Map y_true (strings) to class indices
        try:
            y_true_idx = np.array(
                [classes.index(y) if y in classes else 0 for y in y_true.ravel()]
            )
        except (ValueError, TypeError):
            y_true_idx = np.zeros(len(y_true), dtype=int)
        acc = (pred_idx == y_true_idx).astype(float)
    else:
        conf = np.asarray(y_prob).ravel()
        acc = (np.asarray(y_true).ravel() > 0.5).astype(float)
    if len(conf) == 0:
        return 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        ece += in_bin.sum() * np.abs(acc[in_bin].mean() - conf[in_bin].mean())
    return float(ece / len(conf))
