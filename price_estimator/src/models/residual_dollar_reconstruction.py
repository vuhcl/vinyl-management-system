"""Single reconstruction path: model outputs → log1p(dollar) for metrics / pyfunc."""
from __future__ import annotations

import numpy as np

from .regressor_metrics import pred_log1p_dollar_for_metrics


def pred_log1p_dollar_from_stored_prediction(
    pred_stored: np.ndarray,
    median_price_dollar: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    """
    Convert booster predictions to **log1p(dollar)** space.

    Delegates to ``pred_log1p_dollar_for_metrics`` so pyfunc, tuning, and batch scoring
    cannot drift on residual targets.
    """
    return pred_log1p_dollar_for_metrics(
        pred_stored,
        median_price_dollar,
        target_kind,
    )
