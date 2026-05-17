"""Confidence interval resolution for VinylIQ (quantile heads, residual spread, heuristic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from ..features.vinyliq_features import (
    MAX_LOG_PRICE,
    clamp_ordinals_for_inference,
    grade_delta_scale_params_from_cond,
    scaled_condition_log_adjustment,
)
from ..models.condition_adjustment import default_params
from ..models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
)
from ..models.regressor_constants import (
    CONFIDENCE_CALIBRATION_FILE,
    MANIFEST_FILE,
    REGRESSOR_Q_HIGH_FILE,
    REGRESSOR_Q_LOW_FILE,
)
from ..training.sale_floor_targets import (
    inference_price_suggestion_anchor_usd_for_side,
    inference_residual_anchor_usd,
)

_MIN_PRICE_USD = 0.50


def read_manifest(model_dir: Path) -> dict[str, Any]:
    p = model_dir / MANIFEST_FILE
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def hull_float_usd_then_round(
    lo_raw: float,
    hi_raw: float,
    point_usd: float,
) -> tuple[float, float]:
    """
    Convex hull of {lo, hi, point} in float USD, then round to cents.

    Ensures ``round(lo) <= round(point) <= round(hi)`` after optional cent nudge.
    """
    a = min(lo_raw, hi_raw)
    b = max(lo_raw, hi_raw)
    lo_adj = min(a, b, point_usd)
    hi_adj = max(a, b, point_usd)
    lo_r = round(lo_adj, 2)
    hi_r = round(hi_adj, 2)
    pt_r = round(point_usd, 2)
    if lo_r <= pt_r <= hi_r:
        return lo_r, hi_r
    # Rare cent edge: nudge bounds outward by $0.01
    if pt_r < lo_r:
        lo_r = max(0.0, round(pt_r - 0.01, 2))
    if pt_r > hi_r:
        hi_r = round(pt_r + 0.01, 2)
    if lo_r > hi_r:
        lo_r, hi_r = min(lo_r, hi_r), max(lo_r, hi_r)
    return lo_r, hi_r


def symmetric_interval_rounded(
    point_usd: float,
    half_width: float,
) -> tuple[float, float]:
    lo = max(0.0, point_usd - half_width)
    hi = point_usd + half_width
    lo_r, hi_r = round(lo, 2), round(hi, 2)
    pt_r = round(point_usd, 2)
    if lo_r <= pt_r <= hi_r:
        return lo_r, hi_r
    return hull_float_usd_then_round(lo, hi, point_usd)


def legacy_heuristic_interval(point_usd: float) -> tuple[float, float]:
    spread = max(point_usd * 0.12, 1.0)
    return symmetric_interval_rounded(point_usd, spread)


def residual_price_single_ps_path(
    *,
    ladder_side: Literal["media", "sleeve"],
    logp_raw: float,
    stats: dict[str, Any],
    baseline,
    media_condition: str | None,
    sleeve_condition: str | None,
    media_ord: float,
    sleeve_ord: float,
    cond_params: dict[str, Any],
    release_year: float | None,
    scale_p,
    nm_grade_key: str,
) -> tuple[float, float]:
    anchor_f: float | None = None
    use_ps_path = False
    ps_anchor = inference_price_suggestion_anchor_usd_for_side(
        stats,
        role=ladder_side,
        media_condition=media_condition,
        sleeve_condition=sleeve_condition,
    )
    if ps_anchor is not None and ps_anchor > 0.0:
        anchor_f = float(ps_anchor)
        use_ps_path = True
    if anchor_f is None:
        anchor_f = inference_residual_anchor_usd(
            stats,
            nm_grade_key=nm_grade_key,
        )
        if anchor_f is None or anchor_f <= 0.0:
            mp_b = baseline
            anchor_f = (
                float(mp_b)
                if mp_b is not None and float(mp_b) > 0
                else 0.0
            )
    anchor = float(anchor_f)
    logp = logp_raw + float(np.log1p(max(anchor, 0.0)))
    _dp = default_params()
    anchor_scale = float(anchor) if anchor > 0 else 1.0
    ref_grade_adj = clamp_ordinals_for_inference(
        float(cond_params.get("ref_grade", _dp["ref_grade"])),
        float(cond_params.get("ref_grade", _dp["ref_grade"])),
    )[0]
    media_ord_adj, sleeve_ord_adj = media_ord, sleeve_ord
    if use_ps_path:
        media_ord_adj, sleeve_ord_adj = clamp_ordinals_for_inference(
            ref_grade_adj, ref_grade_adj
        )
    logp_adj = scaled_condition_log_adjustment(
        logp,
        media_ord_adj,
        sleeve_ord_adj,
        base_alpha=float(cond_params.get("alpha", _dp["alpha"])),
        base_beta=float(cond_params.get("beta", _dp["beta"])),
        ref_grade=float(cond_params.get("ref_grade", 8.0)),
        anchor_usd=max(anchor_scale, 1e-6),
        release_year=release_year,
        scale_params=scale_p,
    )
    raw_price = float(np.expm1(np.clip(logp_adj, 0, MAX_LOG_PRICE)))
    price = max(raw_price, _MIN_PRICE_USD)
    return price, anchor


def vinyl_usd_from_logp_raw(
    *,
    model: FittedVinylIQRegressor,
    logp_raw: float,
    use_ps_dual_path: bool,
    stats: dict[str, Any],
    baseline,
    media_condition: str | None,
    sleeve_condition: str | None,
    media_ord: float,
    sleeve_ord: float,
    cond_params: dict[str, Any],
    release_year: float | None,
    scale_p,
    nm_grade_key: str,
) -> tuple[float, float | None]:
    """Mirror ``InferenceService.estimate`` dollar reconstruction for one ``logp_raw``."""
    if use_ps_dual_path:
        pm, am = residual_price_single_ps_path(
            ladder_side="media",
            logp_raw=logp_raw,
            stats=stats,
            baseline=baseline,
            media_condition=media_condition,
            sleeve_condition=sleeve_condition,
            media_ord=media_ord,
            sleeve_ord=sleeve_ord,
            cond_params=cond_params,
            release_year=release_year,
            scale_p=scale_p,
            nm_grade_key=nm_grade_key,
        )
        ps, aa = residual_price_single_ps_path(
            ladder_side="sleeve",
            logp_raw=logp_raw,
            stats=stats,
            baseline=baseline,
            media_condition=media_condition,
            sleeve_condition=sleeve_condition,
            media_ord=media_ord,
            sleeve_ord=sleeve_ord,
            cond_params=cond_params,
            release_year=release_year,
            scale_p=scale_p,
            nm_grade_key=nm_grade_key,
        )
        price = max((float(pm) + float(ps)) / 2.0, _MIN_PRICE_USD)
        residual_anchor = float((float(am) + float(aa)) / 2.0)
        return price, residual_anchor

    anchor = 0.0
    residual_anchor: float | None = None
    if model.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        anchor_f = inference_residual_anchor_usd(
            stats,
            nm_grade_key=nm_grade_key,
        )
        if anchor_f is None or anchor_f <= 0.0:
            mp_b = baseline
            anchor_f = (
                float(mp_b)
                if mp_b is not None and float(mp_b) > 0
                else 0.0
            )
        anchor = float(anchor_f)
        residual_anchor = anchor
        logp = logp_raw + float(np.log1p(max(anchor, 0.0)))
    else:
        logp = logp_raw
        mp2 = stats.get("release_lowest_price")
        if mp2 is not None and float(mp2) > 0:
            anchor = float(mp2)

    anchor_scale = float(anchor) if anchor > 0 else 1.0
    _dp = default_params()
    logp_adj = scaled_condition_log_adjustment(
        logp,
        media_ord,
        sleeve_ord,
        base_alpha=float(cond_params.get("alpha", _dp["alpha"])),
        base_beta=float(cond_params.get("beta", _dp["beta"])),
        ref_grade=float(cond_params.get("ref_grade", 8.0)),
        anchor_usd=max(anchor_scale, 1e-6),
        release_year=release_year,
        scale_params=scale_p,
    )
    price = float(np.expm1(np.clip(logp_adj, 0, MAX_LOG_PRICE)))
    price = max(price, _MIN_PRICE_USD)
    return price, residual_anchor


def try_load_quantile_estimators(
    model_dir: Path,
    manifest: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    qi = manifest.get("quantile_intervals")
    if not isinstance(qi, dict) or not qi.get("enabled"):
        return None, None
    low_name = str(qi.get("lower", REGRESSOR_Q_LOW_FILE))
    high_name = str(qi.get("upper", REGRESSOR_Q_HIGH_FILE))
    low_p = model_dir / low_name
    high_p = model_dir / high_name
    try:
        if low_p.is_file() and high_p.is_file():
            return joblib.load(low_p), joblib.load(high_p)
    except Exception:
        return None, None
    return None, None


def resolve_confidence_interval_bounds(
    *,
    model_dir: Path,
    model: FittedVinylIQRegressor,
    manifest: dict[str, Any],
    calibration: dict[str, Any] | None,
    x_row: np.ndarray,
    logp_raw_point: float,
    price_usd_float: float,
    price_usd_rounded: float,
    use_ps_dual_path: bool,
    stats: dict[str, Any],
    baseline,
    media_condition: str | None,
    sleeve_condition: str | None,
    media_ord: float,
    sleeve_ord: float,
    cond_params: dict[str, Any],
    release_year: float | None,
    nm_grade_key: str,
    q_low: Any | None,
    q_high: Any | None,
    min_half_width_usd: float,
) -> tuple[float, float]:
    """Return ``(interval_low, interval_high)`` rounded to cents."""
    scale_p = grade_delta_scale_params_from_cond(cond_params)
    ensemble = manifest.get("ensemble")
    is_ensemble = isinstance(ensemble, dict) and ensemble.get("enabled")

    def _usd(logp_raw: float) -> float:
        p, _ = vinyl_usd_from_logp_raw(
            model=model,
            logp_raw=logp_raw,
            use_ps_dual_path=use_ps_dual_path,
            stats=stats,
            baseline=baseline,
            media_condition=media_condition,
            sleeve_condition=sleeve_condition,
            media_ord=media_ord,
            sleeve_ord=sleeve_ord,
            cond_params=cond_params,
            release_year=release_year,
            scale_p=scale_p,
            nm_grade_key=nm_grade_key,
        )
        return float(p)

    if is_ensemble:
        hw = _fallback_half_width(calibration, min_half_width_usd)
        if hw is not None:
            return symmetric_interval_rounded(price_usd_float, hw)
        return legacy_heuristic_interval(price_usd_float)

    qi = manifest.get("quantile_intervals")
    use_quantiles = (
        isinstance(qi, dict)
        and qi.get("enabled")
        and q_low is not None
        and q_high is not None
        and model.backend == "xgboost"
    )
    if use_quantiles:
        try:
            lr = float(np.asarray(q_low.predict(x_row), dtype=np.float64).ravel()[0])
            hr = float(np.asarray(q_high.predict(x_row), dtype=np.float64).ravel()[0])
            usd_lo = _usd(lr)
            usd_hi = _usd(hr)
            lo_r, hi_r = hull_float_usd_then_round(usd_lo, usd_hi, price_usd_float)
            return lo_r, hi_r
        except Exception:
            pass

    hw = _fallback_half_width(calibration, min_half_width_usd)
    if hw is not None:
        return symmetric_interval_rounded(price_usd_float, hw)

    return legacy_heuristic_interval(price_usd_float)


def _fallback_half_width(
    calibration: dict[str, Any] | None,
    min_half_width_usd: float,
) -> float | None:
    if not calibration:
        return None
    raw = calibration.get("fallback_half_width_usd")
    if raw is None:
        hold = calibration.get("holdout")
        if isinstance(hold, dict):
            raw = hold.get("half_width_usd")
    try:
        hw = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        hw = None
    if hw is None or hw <= 0:
        return None
    return max(hw, float(min_half_width_usd))


DEFAULT_MIN_HALF_WIDTH_USD = 1.0

__all__ = [
    "CONFIDENCE_CALIBRATION_FILE",
    "DEFAULT_MIN_HALF_WIDTH_USD",
    "REGRESSOR_Q_HIGH_FILE",
    "REGRESSOR_Q_LOW_FILE",
    "hull_float_usd_then_round",
    "legacy_heuristic_interval",
    "read_manifest",
    "resolve_confidence_interval_bounds",
    "symmetric_interval_rounded",
    "try_load_quantile_estimators",
    "vinyl_usd_from_logp_raw",
]
