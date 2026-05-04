"""Dollar-space metrics and diagnostics for VinylIQ regressors (log1p targets)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .regressor_constants import TARGET_KIND_RESIDUAL_LOG_MEDIAN
from .sample_weights import mutually_exclusive_format_bucket_masks


def log1p_dollar_from_residual(
    pred_z: np.ndarray,
    median_price_dollar: np.ndarray,
) -> np.ndarray:
    """``log1p(y_dollar) ≈ pred_z + log1p(median_anchor)``."""
    z = np.asarray(pred_z, dtype=np.float64)
    m = np.maximum(np.asarray(median_price_dollar, dtype=np.float64), 0.0)
    return z + np.log1p(m)


def log1p_dollar_targets_for_metrics(
    y_stored: np.ndarray,
    median_anchors: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    """Convert training targets to log1p(dollar) for MAE/MdAPE when using residual target."""
    y = np.asarray(y_stored, dtype=np.float64)
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        return y + np.log1p(np.maximum(np.asarray(median_anchors, dtype=np.float64), 0.0))
    return y


def pred_log1p_dollar_for_metrics(
    pred_stored: np.ndarray,
    median_anchors: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    """Convert model predictions to log1p(dollar) for metrics and tuning (then ``expm1`` in MAE/MdAPE/WAPE)."""
    p = np.asarray(pred_stored, dtype=np.float64)
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        return p + np.log1p(np.maximum(np.asarray(median_anchors, dtype=np.float64), 0.0))
    return p


def ensemble_blend_weight_log_anchor(
    median_anchor_usd: np.ndarray,
    *,
    center_log1p: float,
    scale: float,
) -> np.ndarray:
    """
    Sigmoid weight for the **NM-substrings** head vs **ordinal-cascade** head.

    ``w -> 1`` as ``log1p(anchor)`` increases past ``center_log1p``; ordinal weight is ``1 - w``.
    """
    m = np.maximum(np.asarray(median_anchor_usd, dtype=np.float64), 0.0)
    lx = np.log1p(m)
    c = float(center_log1p)
    s = max(float(scale), 1e-9)
    z = (lx - c) / s
    return 1.0 / (1.0 + np.exp(-z))


def _dollars_from_log1p(y_true_log1p: np.ndarray, pred_log1p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.expm1(np.asarray(y_true_log1p, dtype=np.float64))
    yp = np.expm1(np.asarray(pred_log1p, dtype=np.float64))
    return yt, yp


def mae_dollars(y_true_log1p: np.ndarray, pred_log1p: np.ndarray) -> float:
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    return float(np.mean(np.abs(yp - yt)))


def wape_dollars(y_true_log1p: np.ndarray, pred_log1p: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error on dollar prices:
    ``sum(|pred - true|) / sum(|true|)``.

    Interprets average error **relative to total dollar volume** of the batch (unlike MAE, which
    weights cheap and expensive records equally in absolute dollars).
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    den = float(np.sum(np.abs(yt)))
    if den <= 0:
        return float("nan")
    return float(np.sum(np.abs(yp - yt)) / den)


def median_ape_dollars(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    *,
    price_floor: float = 1.0,
) -> float:
    """
    Median absolute percentage error: ``median(|pred - true| / max(true, price_floor))``.

    A **relative** per-row error (robust to skew vs mean APE). ``price_floor`` avoids huge ratios
    when the label is near zero.
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    return float(np.median(ape))


def median_ape_train_median_baseline(
    y_train_log1p: np.ndarray,
    y_eval_log1p: np.ndarray,
    *,
    price_floor: float = 1.0,
) -> float:
    """
    Median APE if we predict ``median(y_train)`` in log1p space for every eval row.

    Use to see whether the model beats a trivial constant (if not, focus on features/labels).
    """
    yt_tr = np.asarray(y_train_log1p, dtype=np.float64)
    y_ev = np.asarray(y_eval_log1p, dtype=np.float64)
    mu = float(np.median(yt_tr))
    pred = np.full_like(y_ev, mu, dtype=np.float64)
    return median_ape_dollars(y_ev, pred, price_floor=price_floor)


def metrics_dollar_from_log1p_masked(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    mask: np.ndarray,
    *,
    min_count: int = 15,
) -> tuple[float, float, float]:
    """MAE / WAPE / MdAPE on a boolean row mask; NaNs if ``sum(mask) < min_count``."""
    y = np.asarray(y_true_log1p, dtype=np.float64)
    p = np.asarray(pred_log1p, dtype=np.float64)
    m = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(y)
        & np.isfinite(p)
    )
    if int(np.sum(m)) < int(min_count):
        return (float("nan"), float("nan"), float("nan"))
    return (
        mae_dollars(y[m], p[m]),
        wape_dollars(y[m], p[m]),
        median_ape_dollars(y[m], p[m]),
    )


def true_dollar_quartile_masks(yt: np.ndarray, *, n_bins: int = 4) -> list[np.ndarray]:
    """
    Boolean masks partitioning rows by **true** dollar price ``yt`` (cheap → expensive).

    Same edges as ``median_ape_dollar_quartiles`` / ``median_ape_quartile_format_slice_table``.
    """
    y = np.asarray(yt, dtype=np.float64)
    n = int(n_bins)
    qs = np.linspace(0.0, 1.0, n + 1)
    edges = np.quantile(y, qs)
    q_masks: list[np.ndarray] = []
    for i in range(n):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n - 1:
            q_masks.append((y >= lo) & (y <= hi))
        else:
            q_masks.append((y >= lo) & (y < hi))
    return q_masks


def median_ape_dollar_quartiles(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    *,
    price_floor: float = 1.0,
    n_bins: int = 4,
) -> list[float]:
    """
    Median APE within bins of **true** dollar price (quartiles by default).

    Bin 0 is the cheapest true-price slice; bin ``n_bins - 1`` the most expensive.
    Cheap releases often dominate a single headline MdAPE.
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    n = int(n_bins)
    masks = true_dollar_quartile_masks(yt, n_bins=n)
    out: list[float] = []
    for m in masks:
        if not np.any(m):
            out.append(float("nan"))
        else:
            out.append(float(np.median(ape[m])))
    return out


def weighted_format_median_ape_dollars(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    X: np.ndarray,
    feature_columns: list[str],
    format_weights: dict[str, float],
    *,
    min_count: int = 15,
) -> float:
    """
    Average of per-bucket median APE (log1p dollar space), weighted by ``format_weights``.

    Buckets use the same mutually exclusive masks as slice tables. Buckets with
    fewer than ``min_count`` rows are skipped.
    """
    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    default_w = float(format_weights.get("default", 1.0))
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    num = 0.0
    den = 0.0
    for name in order:
        m = buckets[name]
        cnt = int(np.sum(m))
        if cnt < int(min_count):
            continue
        md = median_ape_dollars(y_true_log1p[m], pred_log1p[m])
        if not np.isfinite(md):
            continue
        wf = float(format_weights.get(name, default_w))
        num += wf * float(md)
        den += wf
    if den <= 0:
        return float("nan")
    return float(num / den)


def median_ape_quartile_format_slice_diagnostics(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    X: np.ndarray,
    feature_columns: list[str],
    *,
    price_floor: float = 1.0,
    n_quartiles: int = 4,
    min_count: int = 15,
) -> list[dict[str, Any]]:
    """
    Per (quartile × format) cell with ``n_rows >= min_count``: median / mean / p90 / max APE.

    Use to sanity-check console lines that show ``0.0%`` (one-decimal formatting can hide small
    non-zero medians; ``max_ape`` reveals heavy tails).
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    q_masks = true_dollar_quartile_masks(yt, n_bins=int(n_quartiles))
    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    out: list[dict[str, Any]] = []
    for qi, qm in enumerate(q_masks):
        for name in order:
            mask = qm & buckets[name]
            cnt = int(np.sum(mask))
            if cnt < int(min_count):
                continue
            a = ape[mask]
            out.append(
                {
                    "quartile": qi,
                    "slice": name,
                    "n_rows": cnt,
                    "median_ape": float(np.median(a)),
                    "mean_ape": float(np.mean(a)),
                    "p90_ape": float(np.percentile(a, 90)),
                    "max_ape": float(np.max(a)),
                }
            )
    return out


def median_ape_quartile_format_slice_table(
    y_true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    X: np.ndarray,
    feature_columns: list[str],
    *,
    price_floor: float = 1.0,
    n_quartiles: int = 4,
    min_count: int = 15,
) -> list[dict[str, Any]]:
    """
    Median APE for each (true-dollar quartile × mutually exclusive format bucket).

    Quartiles match ``median_ape_dollar_quartiles`` (Q1 = cheapest true ``y``).

    **Printing note:** ``100 * median_ape`` at one decimal can show ``0.0%`` when the true
    median APE is below ~0.0005 (0.05%). Use ``median_ape_quartile_format_slice_diagnostics``
    for p90/max when spot-checking.
    """
    yt, yp = _dollars_from_log1p(y_true_log1p, pred_log1p)
    floor = max(float(price_floor), 1e-9)
    den = np.maximum(yt, floor)
    ape = np.abs(yp - yt) / den
    n_bins = int(n_quartiles)
    q_masks = true_dollar_quartile_masks(yt, n_bins=n_bins)

    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    out: list[dict[str, Any]] = []
    for qi, qm in enumerate(q_masks):
        for name in order:
            bm = buckets[name]
            mask = qm & bm
            cnt = int(np.sum(mask))
            if cnt < int(min_count):
                md = float("nan")
            else:
                md = float(np.median(ape[mask]))
            out.append(
                {
                    "quartile": qi,
                    "slice": name,
                    "median_ape": md,
                    "n_rows": cnt,
                }
            )
    return out
