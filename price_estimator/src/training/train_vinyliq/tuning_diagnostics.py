"""Slice / quartile logging helpers for the tuning loop (stderr + optional MLflow)."""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

from ...models.fitted_regressor import (
    metrics_dollar_from_log1p_masked,
    median_ape_quartile_format_slice_diagnostics,
)


def log_slice_metrics_block(
    *,
    split_label: str,
    y_lp: np.ndarray,
    pred_lp: np.ndarray,
    mask_nm: np.ndarray,
    mask_cold: np.ndarray,
    mask_ord: np.ndarray,
    mflow_on: bool,
    mlflow: Any,
    min_count: int = 15,
) -> None:
    """NM-comps, cold-start, and ordinal-comps slices in log1p-dollar space."""
    for name, mask in (
        ("nm_comps", mask_nm),
        ("cold_start_no_nm_comps", mask_cold),
        ("ordinal_comps", mask_ord),
    ):
        mae_s, wape_s, mdape_s = metrics_dollar_from_log1p_masked(
            y_lp, pred_lp, mask, min_count=min_count
        )
        n_m = int(np.sum(mask & np.isfinite(y_lp) & np.isfinite(pred_lp)))
        if math.isnan(mdape_s):
            print(
                f"  {split_label} {name}: n<{min_count} (n={n_m}) — MdAPE skipped",
            )
        else:
            print(
                f"  {split_label} {name}: MAE ${mae_s:.4f} | "
                f"WAPE {100.0 * wape_s:.2f}% | median APE {100.0 * mdape_s:.2f}% "
                f"(n={n_m})",
            )
        if mflow_on:
            mlflow.log_metric(f"{split_label}_{name}_n_rows", float(n_m))
            if not math.isnan(mae_s):
                mlflow.log_metric(f"{split_label}_{name}_mae_dollars_approx", mae_s)
            if not math.isnan(wape_s):
                mlflow.log_metric(f"{split_label}_{name}_wape_dollars", wape_s)
            if not math.isnan(mdape_s):
                mlflow.log_metric(f"{split_label}_{name}_median_ape_dollars", mdape_s)


def slice_metric_debug_enabled() -> bool:
    return os.environ.get("VINYLIQ_SLICE_METRIC_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def log_quartile_format_slice_diagnostics(
    split_label: str,
    y_lp: np.ndarray,
    pred_lp: np.ndarray,
    X_sub: np.ndarray,
    cols: list[str],
) -> None:
    """Stderr table when ``VINYLIQ_SLICE_METRIC_DEBUG`` is set."""
    rows = median_ape_quartile_format_slice_diagnostics(
        y_lp, pred_lp, X_sub, cols, min_count=15
    )
    print(
        f"[VINYLIQ_SLICE_METRIC_DEBUG] {split_label}: quartile×format "
        "(MdAPE / mean / p90 / max as % of true $)",
        file=sys.stderr,
    )
    for r in rows:
        md, mn, p9, mx = (
            r["median_ape"],
            r["mean_ape"],
            r["p90_ape"],
            r["max_ape"],
        )
        print(
            f"  Q{r['quartile'] + 1} {r['slice']:9s} n={r['n_rows']:<5d} "
            f"md={100.0 * md:7.4f}% mean={100.0 * mn:7.4f}% "
            f"p90={100.0 * p9:7.4f}% max={100.0 * mx:7.4f}%",
            file=sys.stderr,
        )
