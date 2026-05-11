"""Holdout residual-spread calibration for VinylIQ confidence intervals (training-side)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .regressor_constants import TARGET_KIND_RESIDUAL_LOG_MEDIAN
from .regressor_metrics import (
    log1p_dollar_targets_for_metrics,
    pred_log1p_dollar_for_metrics,
)


def dollars_abs_errors_from_log1p_pair(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
) -> np.ndarray:
    """Same dollar reconstruction as ``mae_dollars`` before mean — absolute errors per row."""
    yt = np.expm1(np.asarray(y_true_log1p, dtype=np.float64))
    yp = np.expm1(np.asarray(pred_log1p, dtype=np.float64))
    return np.abs(yp - yt)


def compute_holdout_residual_calibration(
    *,
    y_stored: np.ndarray,
    pred_stored: np.ndarray,
    median_anchors: np.ndarray | None,
    target_kind: str,
    abs_error_quantile: float,
    min_half_width_usd: float,
    min_holdout_n: int,
    interval_source_preference: str,
) -> dict[str, Any]:
    """
    Build ``confidence_calibration.json`` payload.

    Dollar errors match :func:`~price_estimator.src.models.regressor_metrics.mae_dollars`
    reconstruction (log1p-dollar space → ``expm1`` → ``abs``).
    """
    y_st = np.asarray(y_stored, dtype=np.float64)
    pr = np.asarray(pred_stored, dtype=np.float64)
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        med = np.asarray(median_anchors, dtype=np.float64)
        y_lp = log1p_dollar_targets_for_metrics(y_st, med, target_kind)
        p_lp = pred_log1p_dollar_for_metrics(pr, med, target_kind)
    else:
        y_lp = y_st
        p_lp = pr
    mask = np.isfinite(y_lp) & np.isfinite(p_lp)
    if not np.any(mask):
        half_width = float(min_half_width_usd)
        med_ae = 0.0
        n = 0
    else:
        abs_err = dollars_abs_errors_from_log1p_pair(y_lp[mask], p_lp[mask])
        n = int(abs_err.shape[0])
        med_ae = float(np.median(abs_err))
        q = float(abs_error_quantile)
        if n < int(min_holdout_n):
            half_width = max(float(med_ae), float(min_half_width_usd))
        else:
            half_width = float(np.quantile(abs_err, q))
            half_width = max(half_width, float(min_half_width_usd))

    return {
        "schema_version": 1,
        "holdout": {
            "n": n,
            "abs_error_quantile_level": float(abs_error_quantile),
            "half_width_usd": half_width,
            "median_abs_error_usd": med_ae,
        },
        "interval_source_preference": str(interval_source_preference),
        "fallback_half_width_usd": half_width,
    }


def write_calibration(model_dir: Path | str, payload: dict[str, Any]) -> None:
    from .regressor_constants import CONFIDENCE_CALIBRATION_FILE

    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / CONFIDENCE_CALIBRATION_FILE).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def calibration_payload_from_log1p_dollar_columns(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    *,
    abs_error_quantile: float,
    min_half_width_usd: float,
    min_holdout_n: int,
    interval_source_preference: str,
) -> dict[str, Any]:
    """Holdout calibration when both arrays are already in log1p-dollar space (expm1 MAE path)."""
    y_lp = np.asarray(y_true_log1p, dtype=np.float64)
    p_lp = np.asarray(pred_log1p, dtype=np.float64)
    mask = np.isfinite(y_lp) & np.isfinite(p_lp)
    if not np.any(mask):
        half_width = float(min_half_width_usd)
        med_ae = 0.0
        n = 0
    else:
        abs_err = dollars_abs_errors_from_log1p_pair(y_lp[mask], p_lp[mask])
        n = int(abs_err.shape[0])
        med_ae = float(np.median(abs_err))
        q = float(abs_error_quantile)
        if n < int(min_holdout_n):
            half_width = max(float(med_ae), float(min_half_width_usd))
        else:
            half_width = float(np.quantile(abs_err, q))
            half_width = max(half_width, float(min_half_width_usd))

    return {
        "schema_version": 1,
        "holdout": {
            "n": n,
            "abs_error_quantile_level": float(abs_error_quantile),
            "half_width_usd": half_width,
            "median_abs_error_usd": med_ae,
        },
        "interval_source_preference": str(interval_source_preference),
        "fallback_half_width_usd": half_width,
    }


def load_calibration(model_dir: Path | str) -> dict[str, Any] | None:
    from .regressor_constants import CONFIDENCE_CALIBRATION_FILE

    p = Path(model_dir) / CONFIDENCE_CALIBRATION_FILE
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None
