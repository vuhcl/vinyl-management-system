"""Optional sample weights and format-bucket multipliers for regressor fitting."""
from __future__ import annotations

import numpy as np


def _feature_column_index(feature_columns: list[str], name: str) -> int | None:
    try:
        return feature_columns.index(name)
    except ValueError:
        return None


def mutually_exclusive_format_bucket_masks(
    X: np.ndarray,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    """
    One bucket per row for slice metrics (priority: box_multi > 7 > 10 > 12 > lp > cd > other).

    Missing format columns (older models) yield all-zero bits → ``other``.
    """
    nrow = int(X.shape[0])

    def col(name: str) -> np.ndarray:
        j = _feature_column_index(feature_columns, name)
        if j is None:
            return np.zeros(nrow, dtype=np.float64)
        return np.asarray(X[:, j], dtype=np.float64)

    c_box = col("is_box_set")
    c_multi = col("is_multi_disc")
    c7 = col("is_7inch")
    c10 = col("is_10inch")
    c12 = col("is_12inch")
    clp = col("is_lp")
    ccd = col("is_cd")

    box_multi = (c_box >= 0.5) | (c_multi >= 0.5)
    seven = (~box_multi) & (c7 >= 0.5)
    ten = (~box_multi) & (~seven) & (c10 >= 0.5)
    twelve = (~box_multi) & (~seven) & (~ten) & (c12 >= 0.5)
    lp = (~box_multi) & (~seven) & (~ten) & (~twelve) & (clp >= 0.5)
    cd = (~box_multi) & (~seven) & (~ten) & (~twelve) & (~lp) & (ccd >= 0.5)
    other = ~(box_multi | seven | ten | twelve | lp | cd)
    return {
        "box_multi": box_multi,
        "seven": seven,
        "ten": ten,
        "twelve": twelve,
        "lp": lp,
        "cd": cd,
        "other": other,
    }


def training_sample_weights_from_anchors(
    anchors: np.ndarray,
    mode: str | None,
) -> np.ndarray | None:
    """
    Optional per-row weights for ``fit_regressor`` (upweight low-dollar anchors).

    Modes: ``None`` / empty / ``off`` → no weights; ``inv_sqrt_anchor`` →
    ``w ∝ 1/sqrt(max(anchor, 1))`` normalized to mean 1.
    """
    if mode is None or not str(mode).strip():
        return None
    key = str(mode).strip().lower()
    if key in ("none", "null", "off", "false", "0", "no"):
        return None
    m = np.maximum(np.asarray(anchors, dtype=np.float64), 1.0)
    if key == "inv_sqrt_anchor":
        w = 1.0 / np.sqrt(m)
        s = float(np.sum(w))
        if s <= 0:
            return None
        w *= float(len(w)) / s
        return w.astype(np.float64)
    raise ValueError(
        f"Unknown tuning.sample_weight mode {mode!r} (use null or inv_sqrt_anchor)"
    )


def combine_anchor_and_format_sample_weights(
    anchors: np.ndarray,
    anchor_mode: str | None,
    X: np.ndarray,
    feature_columns: list[str],
    format_multipliers: dict[str, float] | None,
) -> np.ndarray | None:
    """
    Apply optional per-format multipliers on top of ``training_sample_weights_from_anchors``.

    Final weights are renormalized to mean 1. ``format_multipliers`` maps bucket names
    (``box_multi``, ``seven``, … ``other``) and optional ``default`` for unspecified keys.
    """
    base = training_sample_weights_from_anchors(anchors, anchor_mode)
    return apply_format_multipliers_to_weights(base, X, feature_columns, format_multipliers)


def apply_format_multipliers_to_weights(
    base: np.ndarray | None,
    X: np.ndarray,
    feature_columns: list[str],
    mults: dict[str, float] | None,
) -> np.ndarray | None:
    """Multiply per-row weights by format bucket multipliers; renorm to mean 1."""
    if mults is None or not mults:
        return base
    n = int(X.shape[0])
    if base is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(base, dtype=np.float64).copy()
    default_m = float(mults.get("default", 1.0))
    buckets = mutually_exclusive_format_bucket_masks(X, feature_columns)
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    for name in order:
        mf = float(mults.get(name, default_m))
        if mf != 1.0 and name in buckets:
            w[buckets[name]] *= mf
    s = float(np.sum(w))
    if s <= 0:
        return base
    w *= float(n) / s
    return w.astype(np.float64)
