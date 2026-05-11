"""Shared constants for VinylIQ fitted regressors (targets + on-disk artifact names)."""
from __future__ import annotations

TARGET_KIND_DOLLAR_LOG1P = "dollar_log1p"
TARGET_KIND_RESIDUAL_LOG_MEDIAN = "residual_log_median"

MANIFEST_FILE = "model_manifest.json"
REGRESSOR_FILE = "regressor.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"
TARGET_LOG1P_FILE = "target_log1p.joblib"
LEGACY_XGB_FILE = "xgb_model.joblib"

CONFIDENCE_CALIBRATION_FILE = "confidence_calibration.json"
REGRESSOR_Q_LOW_FILE = "regressor_q_low.joblib"
REGRESSOR_Q_HIGH_FILE = "regressor_q_high.joblib"
