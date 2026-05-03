"""
Cross-grade fit for ``grade_delta_scale.json`` (pooled sale history × anchor × decade).

Joins mirror training: ``release_sale`` + ``marketplace_stats`` anchor via
``pre_uplift_grade_anchor_usd`` + ``releases_features.year``.
"""
from __future__ import annotations

import json
import math
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from price_estimator.src.features.vinyliq_features import (
    GradeDeltaScaleParams,
    condition_string_to_ordinal,
    grade_scale_from_params,
    log1p_nm_equivalent_from_sale_usd,
)
from price_estimator.src.training.sale_floor_targets import (
    parse_iso_datetime,
    pre_uplift_grade_anchor_usd,
    sale_row_usd,
)


def _repo_root() -> Path:
    """Workspace root (``vinyl_management_system``)."""
    return Path(__file__).resolve().parents[3]


def _try_git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_repo_root().parent,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _positive(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def load_marketplace_by_release(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT release_id, release_lowest_price,
                   price_suggestions_json
            FROM marketplace_stats
            """
        )
        out: dict[str, dict[str, Any]] = {}
        for r in cur.fetchall():
            rid = str(r["release_id"])
            out[rid] = {k: r[k] for k in r.keys()}
        return out
    finally:
        conn.close()


def load_year_by_release(path: Path) -> dict[str, float | None]:
    if not path.is_file():
        return {}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        out: dict[str, float | None] = {}
        for r in conn.execute("SELECT release_id, year FROM releases_features"):
            y = r["year"]
            try:
                out[str(r["release_id"])] = float(y) if y is not None else None
            except (TypeError, ValueError):
                out[str(r["release_id"])] = None
        return out
    finally:
        conn.close()


def load_sale_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cols = {x[1] for x in conn.execute("PRAGMA table_info(release_sale)")}
        want = "price_user_usd_approx" in cols
        if want:
            q = (
                "SELECT release_id, order_date, media_condition, sleeve_condition, "
                "price_user_usd_approx FROM release_sale"
            )
        else:
            q = (
                "SELECT release_id, order_date, media_condition, sleeve_condition, "
                "price_original_text, price_user_currency_text FROM release_sale"
            )
        return [dict(r) for r in conn.execute(q)]
    finally:
        conn.close()


def sale_frame_from_dbs(
    sale_history_db: Path,
    marketplace_db: Path,
    feature_store_db: Path,
    *,
    nm_grade_key: str,
) -> pd.DataFrame:
    """One row per sale with ``log_price``, ``eff_ord``, ``anchor_usd``, ``year``."""
    mp = load_marketplace_by_release(marketplace_db)
    years = load_year_by_release(feature_store_db)
    rows_out: list[dict[str, Any]] = []
    for r in load_sale_rows(sale_history_db):
        rid = str(r.get("release_id") or "").strip()
        if not rid:
            continue
        usd = _positive(r.get("price_user_usd_approx"))
        if usd is None:
            usd = sale_row_usd(r)
        if usd is None or usd <= 0:
            continue
        mp_row = mp.get(rid)
        if mp_row is None:
            continue
        anchor = pre_uplift_grade_anchor_usd(dict(mp_row), nm_grade_key=nm_grade_key)
        yr = years.get(rid)
        mo = condition_string_to_ordinal(r.get("media_condition"))
        so = condition_string_to_ordinal(r.get("sleeve_condition"))
        eff = float(min(mo, so))
        if eff < 0:
            continue
        rows_out.append(
            {
                "release_id": rid,
                "order_date": r.get("order_date"),
                "usd": float(usd),
                "log_price": float(math.log1p(float(usd))),
                "media_ord": float(mo),
                "sleeve_ord": float(so),
                "eff_ord": eff,
                "anchor_usd": float(anchor),
                "year": float(yr) if yr is not None and math.isfinite(yr) else np.nan,
            }
        )
    if not rows_out:
        return pd.DataFrame()
    return pd.DataFrame(rows_out)


def add_anchor_decade_bins(
    df: pd.DataFrame,
    *,
    n_anchor_bins: int = 10,
) -> pd.DataFrame:
    """Adds ``anchor_decile`` (0..n-1) and ``decade`` (floor year / 10 * 10; NaN → -9990)."""
    if df.empty:
        return df
    x = np.maximum(df["anchor_usd"].to_numpy(dtype=np.float64), 1e-6)
    try:
        df = df.copy()
        df["anchor_decile"] = pd.qcut(
            x, q=n_anchor_bins, labels=False, duplicates="drop"
        ).astype("float64")
    except ValueError:
        df = df.copy()
        df["anchor_decile"] = 0.0
    # Constant anchors (or too few unique values) can yield NA bins from ``qcut``.
    df["anchor_decile"] = pd.to_numeric(df["anchor_decile"], errors="coerce").fillna(0.0)
    yv = df["year"].to_numpy(dtype=np.float64)
    decade = np.where(np.isfinite(yv), np.floor(yv / 10.0) * 10.0, -9990.0)
    df = df.copy()
    df["decade"] = decade
    df["bin_key"] = (
        df["anchor_decile"].astype(np.int64).astype(str)
        + "_"
        + df["decade"].astype(np.int64).astype(str)
    )
    return df


@dataclass(frozen=True)
class BinDelta:
    bin_key: str
    n_nm: int
    n_vgp: int
    emp_delta_log: float
    med_vgp_usd: float
    med_anchor: float
    med_year: float


def compute_bin_deltas(
    df: pd.DataFrame,
    *,
    min_bin_rows: int,
    min_grade_rows: int,
) -> list[BinDelta]:
    """Per ``bin_key``: ``median(logp|NM) - median(logp|VG+)`` (strict VG+: both grades in ``[6,7)``)."""
    if df.empty:
        return []
    nm = df["eff_ord"] >= 7.0
    # Strict VG+: both media and sleeve in [6, 7), excluding mixed NM sleeve/media rows.
    vgp = (
        (df["media_ord"] >= 6.0)
        & (df["media_ord"] < 7.0)
        & (df["sleeve_ord"] >= 6.0)
        & (df["sleeve_ord"] < 7.0)
    )
    out: list[BinDelta] = []
    for key, part in df.groupby("bin_key", sort=False):
        n = len(part)
        if n < min_bin_rows:
            continue
        p_nm = part.loc[nm.loc[part.index], "log_price"]
        p_vg = part.loc[vgp.loc[part.index], "log_price"]
        if len(p_nm) < min_grade_rows or len(p_vg) < min_grade_rows:
            continue
        med_nm = float(np.median(p_nm.to_numpy(dtype=np.float64)))
        med_vg = float(np.median(p_vg.to_numpy(dtype=np.float64)))
        emp = med_nm - med_vg
        med_vgp_usd = float(np.median(part.loc[vgp.loc[part.index], "usd"].to_numpy(dtype=np.float64)))
        med_anchor = float(np.median(part["anchor_usd"].to_numpy(dtype=np.float64)))
        med_year = float(np.nanmedian(part["year"].to_numpy(dtype=np.float64)))
        if not math.isfinite(med_year):
            med_year = 2000.0
        out.append(
            BinDelta(
                bin_key=str(key),
                n_nm=int(len(p_nm)),
                n_vgp=int(len(p_vg)),
                emp_delta_log=float(emp),
                med_vgp_usd=med_vgp_usd,
                med_anchor=med_anchor,
                med_year=med_year,
            )
        )
    return out


@dataclass(frozen=True)
class BinContrastTri:
    """Per anchor×decade bin: symmetric VG+↔NM gaps plus asymmetric slices for α / β."""

    bin_key: str
    n_nm: int
    n_vgp: int
    emp_sym: float
    med_vgp_usd: float
    med_anchor_sym: float
    med_year_sym: float
    n_media_slice: int
    emp_media: float | None
    med_media_slice_usd: float
    med_anchor_media: float
    med_year_media: float
    n_sleeve_slice: int
    emp_sleeve: float | None
    med_sleeve_slice_usd: float
    med_anchor_sleeve: float
    med_year_sleeve: float


def compute_bin_contrast_triplets(
    df: pd.DataFrame,
    *,
    min_bin_rows: int,
    min_grade_rows: int,
) -> list[BinContrastTri]:
    """
    Extends symmetric NM−VG+ bins with asymmetric strata:

    - **Media slice:** ``media_ord ∈ [6,7)`` and ``sleeve_ord ≥ 7``.
    - **Sleeve slice:** ``media_ord ≥ 7`` and ``sleeve_ord ∈ [6,7)``.

    Empirical gaps: ``median(log|NM) − median(log|slice)`` within each bin.
    """
    if df.empty:
        return []
    nm_m = df["eff_ord"] >= 7.0
    vgp_sym_m = (
        (df["media_ord"] >= 6.0)
        & (df["media_ord"] < 7.0)
        & (df["sleeve_ord"] >= 6.0)
        & (df["sleeve_ord"] < 7.0)
    )
    media_slice_m = (
        (df["media_ord"] >= 6.0)
        & (df["media_ord"] < 7.0)
        & (df["sleeve_ord"] >= 7.0)
    )
    sleeve_slice_m = (
        (df["media_ord"] >= 7.0)
        & (df["sleeve_ord"] >= 6.0)
        & (df["sleeve_ord"] < 7.0)
    )
    out: list[BinContrastTri] = []
    for key, part in df.groupby("bin_key", sort=False):
        if len(part) < min_bin_rows:
            continue
        nm_ix = nm_m.loc[part.index]
        vgp_ix = vgp_sym_m.loc[part.index]
        p_nm = part.loc[nm_ix, "log_price"]
        p_vg = part.loc[vgp_ix, "log_price"]
        if len(p_nm) < min_grade_rows or len(p_vg) < min_grade_rows:
            continue
        med_nm = float(np.median(p_nm.to_numpy(dtype=np.float64)))
        med_vgp = float(np.median(p_vg.to_numpy(dtype=np.float64)))
        emp_sym = med_nm - med_vgp
        med_vgp_usd = float(
            np.median(part.loc[vgp_ix, "usd"].to_numpy(dtype=np.float64))
        )
        med_anchor_sym = float(np.median(part["anchor_usd"].to_numpy(dtype=np.float64)))
        med_year_sym = float(np.nanmedian(part["year"].to_numpy(dtype=np.float64)))
        if not math.isfinite(med_year_sym):
            med_year_sym = 2000.0

        ms_ix = media_slice_m.loc[part.index]
        ss_ix = sleeve_slice_m.loc[part.index]
        n_media = int(part.loc[ms_ix].shape[0])
        n_sleeve = int(part.loc[ss_ix].shape[0])

        emp_media: float | None = None
        med_media_usd = med_vgp_usd
        med_anchor_media = med_anchor_sym
        med_year_media = med_year_sym
        if len(p_nm) >= min_grade_rows and n_media >= min_grade_rows:
            lp_m = part.loc[ms_ix, "log_price"]
            emp_media = med_nm - float(np.median(lp_m.to_numpy(dtype=np.float64)))
            med_media_usd = float(np.median(part.loc[ms_ix, "usd"].to_numpy(dtype=np.float64)))
            med_anchor_media = float(
                np.median(part.loc[ms_ix, "anchor_usd"].to_numpy(dtype=np.float64))
            )
            med_year_media = float(
                np.nanmedian(part.loc[ms_ix, "year"].to_numpy(dtype=np.float64))
            )
            if not math.isfinite(med_year_media):
                med_year_media = 2000.0

        emp_sleeve: float | None = None
        med_sleeve_usd = med_vgp_usd
        med_anchor_sleeve = med_anchor_sym
        med_year_sleeve = med_year_sym
        if len(p_nm) >= min_grade_rows and n_sleeve >= min_grade_rows:
            lp_s = part.loc[ss_ix, "log_price"]
            emp_sleeve = med_nm - float(np.median(lp_s.to_numpy(dtype=np.float64)))
            med_sleeve_usd = float(np.median(part.loc[ss_ix, "usd"].to_numpy(dtype=np.float64)))
            med_anchor_sleeve = float(
                np.median(part.loc[ss_ix, "anchor_usd"].to_numpy(dtype=np.float64))
            )
            med_year_sleeve = float(
                np.nanmedian(part.loc[ss_ix, "year"].to_numpy(dtype=np.float64))
            )
            if not math.isfinite(med_year_sleeve):
                med_year_sleeve = 2000.0

        out.append(
            BinContrastTri(
                bin_key=str(key),
                n_nm=int(len(p_nm)),
                n_vgp=int(len(p_vg)),
                emp_sym=float(emp_sym),
                med_vgp_usd=med_vgp_usd,
                med_anchor_sym=med_anchor_sym,
                med_year_sym=med_year_sym,
                n_media_slice=n_media,
                emp_media=emp_media,
                med_media_slice_usd=med_media_usd,
                med_anchor_media=med_anchor_media,
                med_year_media=med_year_media,
                n_sleeve_slice=n_sleeve,
                emp_sleeve=emp_sleeve,
                med_sleeve_slice_usd=med_sleeve_usd,
                med_anchor_sleeve=med_anchor_sleeve,
                med_year_sleeve=med_year_sleeve,
            )
        )
    return out


def predicted_media_slice_nm_log_lift(
    sale_usd: float,
    *,
    anchor_usd: float,
    release_year: float,
    base_alpha: float,
    base_beta: float,
    scale: GradeDeltaScaleParams,
) -> float:
    """Lift ``(6,7) → (7,7)``: dominated by ``α`` when sleeve already NM."""
    log_before = float(math.log1p(max(sale_usd, 0.0)))
    log_after = log1p_nm_equivalent_from_sale_usd(
        sale_usd,
        6.0,
        7.0,
        7.0,
        7.0,
        base_alpha=base_alpha,
        base_beta=base_beta,
        anchor_usd=anchor_usd,
        release_year=release_year,
        scale_params=scale,
    )
    return float(log_after - log_before)


def predicted_sleeve_slice_nm_log_lift(
    sale_usd: float,
    *,
    anchor_usd: float,
    release_year: float,
    base_alpha: float,
    base_beta: float,
    scale: GradeDeltaScaleParams,
) -> float:
    """Lift ``(7,6) → (7,7)``: dominated by ``β`` when media already NM."""
    log_before = float(math.log1p(max(sale_usd, 0.0)))
    log_after = log1p_nm_equivalent_from_sale_usd(
        sale_usd,
        7.0,
        6.0,
        7.0,
        7.0,
        base_alpha=base_alpha,
        base_beta=base_beta,
        anchor_usd=anchor_usd,
        release_year=release_year,
        scale_params=scale,
    )
    return float(log_after - log_before)


def _g_at(
    anchor_usd: float,
    release_year: float,
    scale: GradeDeltaScaleParams | None,
) -> float:
    return float(
        grade_scale_from_params(max(float(anchor_usd), 1e-6), release_year, scale)
    )


def solve_alpha_beta_for_scale(
    bins: list[BinContrastTri],
    *,
    scale: GradeDeltaScaleParams,
) -> tuple[float, float, float, dict[str, float]]:
    """
    Weighted LS for ``α``, ``β`` using symmetric + asymmetric contrasts per bin.

    Returns ``(alpha, beta, weighted_mse, diagnostics)``.
    """
    s_ss = 0.0
    s_mm = 0.0
    s_bb = 0.0
    r_a = 0.0
    r_b = 0.0
    for b in bins:
        g_sym = _g_at(b.med_anchor_sym, b.med_year_sym, scale)
        w_sym = float(b.n_nm + b.n_vgp)
        s_ss += w_sym * g_sym * g_sym
        r_a += w_sym * g_sym * b.emp_sym
        r_b += w_sym * g_sym * b.emp_sym

        if b.emp_media is not None:
            g_m = _g_at(b.med_anchor_media, b.med_year_media, scale)
            w_m = float(b.n_nm + b.n_media_slice)
            s_mm += w_m * g_m * g_m
            r_a += w_m * g_m * b.emp_media
        if b.emp_sleeve is not None:
            g_s = _g_at(b.med_anchor_sleeve, b.med_year_sleeve, scale)
            w_s = float(b.n_nm + b.n_sleeve_slice)
            s_bb += w_s * g_s * g_s
            r_b += w_s * g_s * b.emp_sleeve

    p_mat = s_ss + s_mm
    q_mat = s_ss
    r_mat = s_ss
    s_mat = s_ss + s_bb
    det = p_mat * s_mat - q_mat * r_mat
    if abs(det) < 1e-18:
        raise ValueError(
            "singular alpha/beta normal equations (insufficient contrast diversity)",
        )
    alpha = (r_a * s_mat - q_mat * r_b) / det
    beta = (p_mat * r_b - r_a * r_mat) / det

    mse = _weighted_mse_alpha_beta(bins, alpha=alpha, beta=beta, scale=scale)
    diag = {
        "n_bins_symmetric": float(len(bins)),
        "n_bins_asym_media": float(sum(1 for b in bins if b.emp_media is not None)),
        "n_bins_asym_sleeve": float(sum(1 for b in bins if b.emp_sleeve is not None)),
    }
    return float(alpha), float(beta), float(mse), diag


def solve_alpha_beta_ratio_fallback(
    bins: list[BinContrastTri],
    *,
    scale: GradeDeltaScaleParams,
    beta_per_alpha: float,
) -> tuple[float, float, float]:
    """Fit ``s = α+β`` from symmetric bins only; split with ``β = ratio·α``."""
    num = 0.0
    den = 0.0
    for b in bins:
        g_sym = _g_at(b.med_anchor_sym, b.med_year_sym, scale)
        w_sym = float(b.n_nm + b.n_vgp)
        num += w_sym * g_sym * b.emp_sym
        den += w_sym * g_sym * g_sym
    if den <= 0:
        raise ValueError("no symmetric bins for alpha/beta fallback")
    s_sum = num / den
    denom = 1.0 + float(beta_per_alpha)
    alpha = float(s_sum / denom)
    beta = float(beta_per_alpha * alpha)
    mse = _weighted_mse_alpha_beta(bins, alpha=alpha, beta=beta, scale=scale)
    return alpha, beta, mse


def _weighted_mse_alpha_beta(
    bins: list[BinContrastTri],
    *,
    alpha: float,
    beta: float,
    scale: GradeDeltaScaleParams,
) -> float:
    tot = 0.0
    wtot = 0.0
    for b in bins:
        g_sym = _g_at(b.med_anchor_sym, b.med_year_sym, scale)
        w_sym = float(b.n_nm + b.n_vgp)
        pred_sym = float(g_sym) * (alpha + beta)
        tot += w_sym * (b.emp_sym - pred_sym) ** 2
        wtot += w_sym
        if b.emp_media is not None:
            g_m = _g_at(b.med_anchor_media, b.med_year_media, scale)
            w_m = float(b.n_nm + b.n_media_slice)
            pred_m = float(g_m) * alpha
            tot += w_m * (b.emp_media - pred_m) ** 2
            wtot += w_m
        if b.emp_sleeve is not None:
            g_s = _g_at(b.med_anchor_sleeve, b.med_year_sleeve, scale)
            w_s = float(b.n_nm + b.n_sleeve_slice)
            pred_s = float(g_s) * beta
            tot += w_s * (b.emp_sleeve - pred_s) ** 2
            wtot += w_s
    return float(tot / max(wtot, 1.0))


def predicted_vgp_to_nm_log_lift(
    sale_usd: float,
    *,
    anchor_usd: float,
    release_year: float,
    base_alpha: float,
    base_beta: float,
    scale: GradeDeltaScaleParams,
) -> float:
    """``log1p_nm_equiv - log1p(sale)`` for VG+ (6,6) → NM (7,7), same law as training uplift."""
    log_before = float(math.log1p(max(sale_usd, 0.0)))
    log_after = log1p_nm_equivalent_from_sale_usd(
        sale_usd,
        6.0,
        6.0,
        7.0,
        7.0,
        base_alpha=base_alpha,
        base_beta=base_beta,
        anchor_usd=anchor_usd,
        release_year=release_year,
        scale_params=scale,
    )
    return float(log_after - log_before)


def grid_score_bin_medians(
    bins: list[BinDelta],
    *,
    base_alpha: float,
    base_beta: float,
    price_ref_usd: float,
    age_center_year: float,
    price_gamma: float,
    age_k: float,
    price_scale_min: float,
    price_scale_max: float,
) -> float:
    """Mean squared error between empirical log lift and model lift (magnitude match)."""
    sp = GradeDeltaScaleParams(
        price_ref_usd=price_ref_usd,
        price_gamma=price_gamma,
        price_scale_min=price_scale_min,
        price_scale_max=price_scale_max,
        age_k=age_k,
        age_center_year=age_center_year,
    )
    if not bins:
        return float("inf")
    se = 0.0
    w = 0.0
    for b in bins:
        pred = predicted_vgp_to_nm_log_lift(
            b.med_vgp_usd,
            anchor_usd=b.med_anchor,
            release_year=b.med_year,
            base_alpha=base_alpha,
            base_beta=base_beta,
            scale=sp,
        )
        diff = b.emp_delta_log - pred
        wi = float(b.n_nm + b.n_vgp)
        se += wi * diff * diff
        w += wi
    return float(se / max(w, 1.0))


def fit_grade_delta_scale_from_frame(
    df: pd.DataFrame,
    *,
    nm_grade_key: str,
    base_alpha: float = 0.06,
    base_beta: float = 0.04,
    fit_alpha_beta: bool = True,
    beta_per_alpha_fallback: float | None = None,
    min_bin_rows: int = 30,
    min_grade_rows: int = 5,
    price_scale_min: float = 0.25,
    price_scale_max: float = 4.0,
    gammas: Iterable[float] | None = None,
    age_ks: Iterable[float] | None = None,
) -> dict[str, Any]:
    """
    Grid search ``price_gamma`` and ``age_k``; ``price_ref_usd`` / ``age_center_year`` from medians.

    When ``fit_alpha_beta`` is True (default), jointly estimates ``α`` and ``β`` from pooled
    symmetric NM−VG+ gaps plus **asymmetric** strata (VG+ media / NM sleeve vs NM media / VG+
    sleeve), then picks the ``(γ, age_k)`` grid point minimizing weighted MSE across all three.

    When ``fit_alpha_beta`` is False, keeps legacy behavior: fixed ``base_alpha`` / ``base_beta``
    and score only the symmetric VG+→NM contrast.

    Optional output keys ``alpha`` / ``beta`` are written when ``fit_alpha_beta`` is True so
    ``load_params_with_grade_delta_overlays`` can merge them into serving ``condition_params``.
    """
    if df.empty:
        raise ValueError("sale frame is empty (check DB paths and joins)")
    ratio_fb = (
        float(beta_per_alpha_fallback)
        if beta_per_alpha_fallback is not None
        else float(base_beta / base_alpha)
        if base_alpha not in (0.0, -0.0)
        else 2.0 / 3.0
    )
    df_b = add_anchor_decade_bins(df)
    bins_legacy = compute_bin_deltas(
        df_b, min_bin_rows=min_bin_rows, min_grade_rows=min_grade_rows
    )
    triplets = compute_bin_contrast_triplets(
        df_b, min_bin_rows=min_bin_rows, min_grade_rows=min_grade_rows
    )
    anchors = df_b["anchor_usd"].to_numpy(dtype=np.float64)
    years = df_b["year"].to_numpy(dtype=np.float64)
    price_ref = float(np.nanmedian(anchors)) if np.any(np.isfinite(anchors)) else 50.0
    age_center = float(np.nanmedian(years[np.isfinite(years)])) if np.any(np.isfinite(years)) else 2000.0
    if not math.isfinite(price_ref) or price_ref <= 0:
        price_ref = 50.0
    if not math.isfinite(age_center):
        age_center = 2000.0

    g_list = tuple(gammas) if gammas is not None else (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    a_list = tuple(age_ks) if age_ks is not None else (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1)

    best_mse = float("inf")
    best_gamma = 0.0
    best_age_k = 0.0
    best_alpha = float(base_alpha)
    best_beta = float(base_beta)
    last_ab_diag: dict[str, float] = {}

    for g in g_list:
        for ak in a_list:
            sp = GradeDeltaScaleParams(
                price_ref_usd=price_ref,
                price_gamma=float(g),
                price_scale_min=price_scale_min,
                price_scale_max=price_scale_max,
                age_k=float(ak),
                age_center_year=age_center,
            )
            if fit_alpha_beta and triplets:
                try:
                    a_hat, b_hat, mse_joint, diag = solve_alpha_beta_for_scale(
                        triplets,
                        scale=sp,
                    )
                except ValueError:
                    a_hat, b_hat, mse_joint = solve_alpha_beta_ratio_fallback(
                        triplets,
                        scale=sp,
                        beta_per_alpha=ratio_fb,
                    )
                    diag = {
                        "n_bins_symmetric": float(len(triplets)),
                        "n_bins_asym_media": float(
                            sum(1 for b in triplets if b.emp_media is not None)
                        ),
                        "n_bins_asym_sleeve": float(
                            sum(1 for b in triplets if b.emp_sleeve is not None)
                        ),
                        "alpha_beta_fallback": 1.0,
                    }
                if mse_joint < best_mse:
                    best_mse = mse_joint
                    best_gamma = float(g)
                    best_age_k = float(ak)
                    best_alpha = a_hat
                    best_beta = b_hat
                    last_ab_diag = diag
            else:
                sc = grid_score_bin_medians(
                    bins_legacy,
                    base_alpha=base_alpha,
                    base_beta=base_beta,
                    price_ref_usd=price_ref,
                    age_center_year=age_center,
                    price_gamma=float(g),
                    age_k=float(ak),
                    price_scale_min=price_scale_min,
                    price_scale_max=price_scale_max,
                )
                if sc < best_mse:
                    best_mse = sc
                    best_gamma = float(g)
                    best_age_k = float(ak)
                    best_alpha = float(base_alpha)
                    best_beta = float(base_beta)
                    last_ab_diag = {}

    od = df["order_date"] if "order_date" in df.columns else None
    parsed_dates: list[datetime] = []
    if od is not None:
        for x in od.tolist():
            if x is None or not str(x).strip():
                continue
            d = parse_iso_datetime(str(x))
            if d is not None:
                parsed_dates.append(d)

    fit_kind = (
        "cross_grade_bin_median_v2_alpha_beta"
        if fit_alpha_beta and triplets
        else "cross_grade_bin_median_v1"
    )
    meta = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "fit_kind": fit_kind,
        "nm_grade_key": nm_grade_key,
        "row_count_sales": int(len(df)),
        "bins_used": len(bins_legacy),
        "triplet_bins_used": len(triplets),
        "min_bin_rows": int(min_bin_rows),
        "min_grade_rows": int(min_grade_rows),
        "base_alpha": float(base_alpha),
        "base_beta": float(base_beta),
        "fit_alpha_beta": bool(fit_alpha_beta),
        "weighted_mse_log_lift": float(best_mse),
        "beta_per_alpha_fallback": float(ratio_fb),
        "git_revision": _try_git_revision(),
    }
    if fit_alpha_beta and triplets:
        meta["fitted_alpha"] = float(best_alpha)
        meta["fitted_beta"] = float(best_beta)
        meta.update({f"ab_diag_{k}": v for k, v in last_ab_diag.items()})
    if parsed_dates:
        meta["sale_date_min"] = min(parsed_dates).date().isoformat()
        meta["sale_date_max"] = max(parsed_dates).date().isoformat()
    if bins_legacy:
        meta["sale_anchor_usd_median"] = float(np.median([b.med_anchor for b in bins_legacy]))
        meta["empirical_delta_log_median_across_bins"] = float(
            np.median([b.emp_delta_log for b in bins_legacy])
        )

    out: dict[str, Any] = {
        "schema_version": 1,
        "fit_metadata": meta,
        "price_ref_usd": price_ref,
        "price_gamma": best_gamma,
        "price_scale_min": float(price_scale_min),
        "price_scale_max": float(price_scale_max),
        "age_k": best_age_k,
        "age_center_year": float(age_center),
    }
    if fit_alpha_beta and triplets:
        out["alpha"] = float(best_alpha)
        out["beta"] = float(best_beta)
    return out
