"""Authoritative drift metrics (PSI, KS, chi-square, proportion z-test) for VinylIQ monitoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# PSI bins: strictly increasing year edges (stable across compares).
_YEAR_EDGES = np.linspace(1899.0, 2041.0, num=31, dtype=float)
_LABEL_TIER_EDGES = np.arange(-0.5, 12.5, 1.0)


def _pct_hist(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Histogram percentages on shared bins (drop NaNs)."""
    ref = np.asarray(ref, dtype=float).ravel()
    cur = np.asarray(cur, dtype=float).ravel()
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("empty reference or current after NaN drop")
    rh, _ = np.histogram(ref, bins=edges)
    ch, _ = np.histogram(cur, bins=edges)
    rsum, csum = rh.sum(), ch.sum()
    if rsum == 0 or csum == 0:
        raise ValueError("zero mass in reference or current histogram")
    return rh.astype(float) / rsum, ch.astype(float) / csum


def population_stability_index(ref_pct: np.ndarray, cur_pct: np.ndarray, eps: float = 1e-6) -> float:
    """Standard PSI on aligned probability vectors."""
    r = np.clip(ref_pct, eps, 1.0)
    c = np.clip(cur_pct, eps, 1.0)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric_year(ref: pd.Series, cur: pd.Series) -> float:
    rp, cp = _pct_hist(ref.to_numpy(dtype=float), cur.to_numpy(dtype=float), _YEAR_EDGES)
    return population_stability_index(rp, cp)


def psi_numeric_decade(ref: pd.Series, cur: pd.Series) -> float:
    rp, cp = _pct_hist(ref.to_numpy(dtype=float), cur.to_numpy(dtype=float), _YEAR_EDGES)
    return population_stability_index(rp, cp)


def psi_numeric_label_tier(ref: pd.Series, cur: pd.Series) -> float:
    rp, cp = _pct_hist(ref.to_numpy(dtype=float), cur.to_numpy(dtype=float), _LABEL_TIER_EDGES)
    return population_stability_index(rp, cp)


def ks_two_sample(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    """Return (statistic, pvalue) from scipy KS two-sample on non-null values."""
    r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(dtype=float)
    c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(dtype=float)
    if r.size == 0 or c.size == 0:
        raise ValueError("KS: empty reference or current")
    res = stats.ks_2samp(r, c, method="auto")
    return float(res.statistic), float(res.pvalue)


def categorical_psi_topk(ref: pd.Series, cur: pd.Series, k: int = 15) -> float:
    """PSI on reference top-K categories plus an ``other`` bucket."""
    ref_s = ref.astype(str).fillna("__na__")
    cur_s = cur.astype(str).fillna("__na__")
    top_cats = ref_s.value_counts().nlargest(k).index.tolist()

    def bucket_pct(s: pd.Series, cats: list[str]) -> np.ndarray:
        parts = [float((s == c).sum()) for c in cats]
        parts.append(float((~s.isin(cats)).sum()))
        a = np.array(parts, dtype=float)
        return a / a.sum()

    rp = bucket_pct(ref_s, top_cats)
    cp = bucket_pct(cur_s, top_cats)
    return population_stability_index(rp, cp)


def chi_square_independence(ref: pd.Series, cur: pd.Series, k: int = 15) -> tuple[float, float]:
    """Chi-square test: ref vs cur counts over reference top-K + other."""
    ref_s = ref.astype(str).fillna("__na__")
    cur_s = cur.astype(str).fillna("__na__")
    top_cats = ref_s.value_counts().nlargest(k).index.tolist()

    def counts_row(s: pd.Series, cats: list[str]) -> list[float]:
        row = [float((s == c).sum()) for c in cats]
        row.append(float((~s.isin(cats)).sum()))
        return row

    obs = np.array(
        [counts_row(ref_s, top_cats), counts_row(cur_s, top_cats)],
        dtype=float,
    )
    if obs.sum(axis=1).min() == 0:
        raise ValueError("chi-square: empty ref or cur column")
    chi2, p, *_ = stats.chi2_contingency(obs + 1e-9)
    return float(chi2), float(p)


def binary_two_sample_psi_and_ztest(
    ref: pd.Series, cur: pd.Series
) -> tuple[float, float]:
    """Two-bin PSI on Bernoulli proportions + two-proportion z-test p-value."""
    r = pd.to_numeric(ref, errors="coerce").fillna(0).astype(int).clip(0, 1)
    c = pd.to_numeric(cur, errors="coerce").fillna(0).astype(int).clip(0, 1)
    n_r, n_c = len(r), len(c)
    p_r, p_c = float(r.mean()), float(c.mean())
    ref_pct = np.array([1 - p_r, p_r], dtype=float)
    cur_pct = np.array([1 - p_c, p_c], dtype=float)
    psi = population_stability_index(ref_pct, cur_pct)
    counts = np.array([r.sum(), c.sum()])
    nobs = np.array([n_r, n_c])
    zstat, pval = proportions_ztest(counts, nobs)
    return psi, float(pval)


@dataclass(frozen=True)
class NumericDriftResult:
    psi: float
    ks_statistic: float
    ks_pvalue: float


def drift_numeric_column(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    column: str,
    *,
    kind: str,
) -> NumericDriftResult:
    """Compute PSI (kind-specific bins) and KS for one numeric column."""
    if column not in ref.columns or column not in cur.columns:
        raise KeyError(column)
    rs, cs = ref[column], cur[column]
    if kind == "year":
        psi = psi_numeric_year(rs, cs)
    elif kind == "decade":
        psi = psi_numeric_decade(rs, cs)
    elif kind == "label_tier":
        psi = psi_numeric_label_tier(rs, cs)
    elif kind == "pred_log1p_usd":
        # Shared quantile bins from reference for dollar/log targets
        rnv = pd.to_numeric(rs, errors="coerce").dropna()
        cnv = pd.to_numeric(cs, errors="coerce").dropna()
        if len(rnv) < 2:
            raise ValueError("pred_log1p_usd: insufficient reference")
        qs = np.linspace(0, 1, 11)
        edges = np.unique(np.quantile(rnv, qs))
        if edges.size < 2:
            edges = np.array([float(rnv.min()), float(rnv.max()) + 1e-6])
        rp, cp = _pct_hist(rnv.to_numpy(), cnv.to_numpy(), edges)
        psi = population_stability_index(rp, cp)
    else:
        raise ValueError(f"unknown numeric kind {kind!r}")
    d, p = ks_two_sample(rs, cs)
    return NumericDriftResult(psi=psi, ks_statistic=float(d), ks_pvalue=float(p))


def drift_predictions(ref: pd.Series, cur: pd.Series) -> NumericDriftResult:
    """Drift on ``pred_log1p_usd`` column."""
    return drift_numeric_column(
        pd.DataFrame({"pred_log1p_usd": ref}),
        pd.DataFrame({"pred_log1p_usd": cur}),
        "pred_log1p_usd",
        kind="pred_log1p_usd",
    )
