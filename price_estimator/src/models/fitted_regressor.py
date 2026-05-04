"""Unified fitted regressor for VinylIQ (log1p target) across boosting backends.

Public API is re-exported from focused modules: ``regressor_fitted`` (wrapper),
``regressor_training`` (fit), ``regressor_metrics`` (dollar metrics), ``sample_weights``.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from .regressor_constants import (
    FEATURE_COLUMNS_FILE,
    LEGACY_XGB_FILE,
    MANIFEST_FILE,
    REGRESSOR_FILE,
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    TARGET_LOG1P_FILE,
)
from .regressor_fitted import FittedVinylIQRegressor
from .regressor_metrics import (
    ensemble_blend_weight_log_anchor,
    log1p_dollar_from_residual,
    log1p_dollar_targets_for_metrics,
    mae_dollars,
    median_ape_dollar_quartiles,
    median_ape_dollars,
    median_ape_quartile_format_slice_diagnostics,
    median_ape_quartile_format_slice_table,
    median_ape_train_median_baseline,
    metrics_dollar_from_log1p_masked,
    pred_log1p_dollar_for_metrics,
    true_dollar_quartile_masks,
    wape_dollars,
    weighted_format_median_ape_dollars,
)
from .regressor_training import fit_regressor, refit_champion
from .sample_weights import (
    apply_format_multipliers_to_weights,
    combine_anchor_and_format_sample_weights,
    mutually_exclusive_format_bucket_masks,
    training_sample_weights_from_anchors,
)


def load_fitted_regressor(directory: Path | str) -> FittedVinylIQRegressor | None:
    """Load manifest bundle, or legacy XGB-only artifact layout."""
    d = Path(directory)
    mf = d / MANIFEST_FILE
    if mf.is_file():
        try:
            manifest = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        backend = str(manifest.get("backend", "")).strip()
        if not backend or not (d / REGRESSOR_FILE).is_file():
            return None
        schema = int(manifest.get("schema_version", 1))
        tk = str(manifest.get("target_kind", "")).strip()
        if schema >= 2 and tk:
            target_kind = tk
        else:
            target_kind = TARGET_KIND_DOLLAR_LOG1P
        tw = bool(joblib.load(d / TARGET_LOG1P_FILE))
        if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            tw = False
        return FittedVinylIQRegressor(
            backend=backend,
            estimator=joblib.load(d / REGRESSOR_FILE),
            feature_columns=list(joblib.load(d / FEATURE_COLUMNS_FILE)),
            target_was_log1p=tw,
            target_kind=target_kind,
        )
    if (d / LEGACY_XGB_FILE).is_file() and (d / FEATURE_COLUMNS_FILE).is_file():
        from .xgb_vinyliq import XGBVinylIQModel

        legacy = XGBVinylIQModel.load(d)
        return FittedVinylIQRegressor(
            backend="xgboost",
            estimator=legacy.model_,
            feature_columns=list(legacy.feature_columns_),
            target_was_log1p=bool(legacy.target_was_log1p_),
            target_kind=TARGET_KIND_DOLLAR_LOG1P,
        )
    return None


__all__ = [
    "FittedVinylIQRegressor",
    "TARGET_KIND_DOLLAR_LOG1P",
    "TARGET_KIND_RESIDUAL_LOG_MEDIAN",
    "MANIFEST_FILE",
    "REGRESSOR_FILE",
    "FEATURE_COLUMNS_FILE",
    "TARGET_LOG1P_FILE",
    "LEGACY_XGB_FILE",
    "load_fitted_regressor",
    "fit_regressor",
    "refit_champion",
    "log1p_dollar_from_residual",
    "log1p_dollar_targets_for_metrics",
    "pred_log1p_dollar_for_metrics",
    "ensemble_blend_weight_log_anchor",
    "metrics_dollar_from_log1p_masked",
    "mae_dollars",
    "wape_dollars",
    "median_ape_dollars",
    "median_ape_train_median_baseline",
    "median_ape_dollar_quartiles",
    "training_sample_weights_from_anchors",
    "mutually_exclusive_format_bucket_masks",
    "combine_anchor_and_format_sample_weights",
    "apply_format_multipliers_to_weights",
    "weighted_format_median_ape_dollars",
    "true_dollar_quartile_masks",
    "median_ape_quartile_format_slice_diagnostics",
    "median_ape_quartile_format_slice_table",
]
