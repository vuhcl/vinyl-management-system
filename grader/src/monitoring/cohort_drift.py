"""Cohort drift between the in-distribution held-out split and the
low-detail (test_thin) cohort.

This measures *cohort shift* on offline data. It is not production traffic
monitoring: there is no live traffic behind the classifier.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from price_estimator.src.monitoring.drift_stats import (
    categorical_psi_topk,
    chi_square_independence,
    ks_two_sample,
    population_stability_index,
)
import numpy as np

CATEGORICAL = ("true_sleeve", "true_media", "pred_sleeve", "pred_media")
NUMERIC = ("text_len", "sleeve_conf", "media_conf")


def _numeric_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(float)
    c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(float)
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    rp = np.histogram(r, edges)[0].astype(float)
    cp = np.histogram(c, edges)[0].astype(float)
    return population_stability_index(rp / rp.sum(), cp / cp.sum())


def cohort_drift(
    ref_rows: list[dict], cur_rows: list[dict], cfg: dict[str, Any]
) -> dict[str, Any]:
    ref, cur = pd.DataFrame(ref_rows), pd.DataFrame(cur_rows)
    cat, num = cfg["categorical"], cfg["numeric"]
    results, flagged = {}, []

    for col in CATEGORICAL:
        if col not in ref or col not in cur:
            continue
        psi = categorical_psi_topk(ref[col], cur[col], k=int(cat["top_k"]))
        chi2, p = chi_square_independence(ref[col], cur[col], k=int(cat["top_k"]))
        drift = psi > float(cat["max_psi_topk"]) and p < float(cat["min_chi2_pvalue"])
        results[col] = {"psi": round(psi, 4), "chi2_p": float(f"{p:.3g}"), "drift": drift}
        if drift:
            flagged.append(col)

    for col in NUMERIC:
        if col not in ref or col not in cur:
            continue
        psi = _numeric_psi(ref[col], cur[col])
        ks, p = ks_two_sample(ref[col], cur[col])
        drift = psi > float(num["max_psi"]) and p < float(num["min_ks_pvalue"])
        results[col] = {"psi": round(psi, 4), "ks": round(ks, 4), "ks_p": float(f"{p:.3g}"), "drift": drift}
        if drift:
            flagged.append(col)

    return {
        "n_reference": len(ref),
        "n_current": len(cur),
        "features": results,
        "flagged": flagged,
        "drift_detected": bool(flagged),
    }
