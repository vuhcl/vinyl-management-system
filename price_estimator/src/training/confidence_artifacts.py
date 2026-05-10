"""Persist confidence calibration (+ optional XGB quantile heads) after VinylIQ training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..models.confidence_calibration import (
    calibration_payload_from_log1p_dollar_columns,
    compute_holdout_residual_calibration,
    write_calibration,
)
from ..models.fitted_regressor import TARGET_KIND_RESIDUAL_LOG_MEDIAN, FittedVinylIQRegressor
from ..models.manifest_merge import merge_model_manifest
from ..models.regressor_constants import REGRESSOR_Q_HIGH_FILE, REGRESSOR_Q_LOW_FILE
from ..models.regressor_training import fit_regressor
from .train_vinyliq.training_config import confidence_interval_settings_from_vinyliq


def write_confidence_training_bundle(
    *,
    model_dir: Path,
    vinyliq_cfg: dict[str, Any],
    target_kind: str,
    champion: FittedVinylIQRegressor | None,
    cols: list[str],
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    median_test: np.ndarray | None,
    ensemble_active: bool,
    log1p_y_test: np.ndarray | None,
    log1p_pred_test: np.ndarray | None,
) -> None:
    """
    Always writes ``confidence_calibration.json``. Optionally fits quantile XGB heads
    (single residual champion only).
    """
    ci = confidence_interval_settings_from_vinyliq(vinyliq_cfg)
    md = Path(model_dir)

    if ensemble_active and log1p_y_test is not None and log1p_pred_test is not None:
        payload = calibration_payload_from_log1p_dollar_columns(
            log1p_y_test,
            log1p_pred_test,
            abs_error_quantile=ci["residual_abs_error_quantile"],
            min_half_width_usd=ci["min_half_width_usd"],
            min_holdout_n=ci["min_holdout_n"],
            interval_source_preference="residual_spread_only",
        )
        write_calibration(md, payload)
        return

    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        assert median_test is not None
        payload = compute_holdout_residual_calibration(
            y_stored=y_test,
            pred_stored=pred_test,
            median_anchors=median_test,
            target_kind=target_kind,
            abs_error_quantile=ci["residual_abs_error_quantile"],
            min_half_width_usd=ci["min_half_width_usd"],
            min_holdout_n=ci["min_holdout_n"],
            interval_source_preference="residual_spread_only",
        )
    else:
        payload = calibration_payload_from_log1p_dollar_columns(
            y_test,
            pred_test,
            abs_error_quantile=ci["residual_abs_error_quantile"],
            min_half_width_usd=ci["min_half_width_usd"],
            min_holdout_n=ci["min_holdout_n"],
            interval_source_preference="residual_spread_only",
        )

    wrote_quantiles = False
    if (
        ci["enabled"]
        and not ensemble_active
        and champion is not None
        and target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN
        and champion.backend == "xgboost"
    ):
        base_p = champion.estimator.get_params()
        for k in ("objective", "quantile_alpha", "callbacks"):
            base_p.pop(k, None)
        ov = ci["xgboost_overrides"]
        n_trees = max(1, int(champion.estimator.get_booster().num_boosted_rounds()))
        low_p = {
            **base_p,
            **ov,
            "objective": "reg:quantileerror",
            "quantile_alpha": ci["lower_alpha"],
            "n_estimators": n_trees,
        }
        high_p = {
            **base_p,
            **ov,
            "objective": "reg:quantileerror",
            "quantile_alpha": ci["upper_alpha"],
            "n_estimators": n_trees,
        }

        fit_kw: dict[str, Any] = dict(random_state=seed, target_kind=target_kind)
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        low_reg, _ = fit_regressor(
            "xgboost",
            low_p,
            x_train,
            y_train,
            cols,
            **fit_kw,
        )
        high_reg, _ = fit_regressor(
            "xgboost",
            high_p,
            x_train,
            y_train,
            cols,
            **fit_kw,
        )
        joblib.dump(low_reg.estimator, md / REGRESSOR_Q_LOW_FILE)
        joblib.dump(high_reg.estimator, md / REGRESSOR_Q_HIGH_FILE)
        merge_model_manifest(
            md,
            {
                "quantile_intervals": {
                    "enabled": True,
                    "lower": REGRESSOR_Q_LOW_FILE,
                    "upper": REGRESSOR_Q_HIGH_FILE,
                    "lower_alpha": ci["lower_alpha"],
                    "upper_alpha": ci["upper_alpha"],
                }
            },
        )
        payload["interval_source_preference"] = "quantile_artifacts"
        wrote_quantiles = True

    if not wrote_quantiles:
        payload["interval_source_preference"] = payload.get(
            "interval_source_preference", "residual_spread_only"
        )

    write_calibration(md, payload)
