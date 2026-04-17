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
            SELECT release_id, median_price, lowest_price, release_lowest_price,
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
    """Per ``bin_key``: ``median(logp|NM) - median(logp|VG+)`` (VG+ = ``6 <= eff_ord < 7``)."""
    if df.empty:
        return []
    nm = df["eff_ord"] >= 7.0
    vgp = (df["eff_ord"] >= 6.0) & (df["eff_ord"] < 7.0)
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
    base_alpha: float = -0.06,
    base_beta: float = -0.04,
    min_bin_rows: int = 30,
    min_grade_rows: int = 5,
    price_scale_min: float = 0.25,
    price_scale_max: float = 4.0,
    gammas: Iterable[float] | None = None,
    age_ks: Iterable[float] | None = None,
) -> dict[str, Any]:
    """
    Grid search ``price_gamma`` and ``age_k``; ``price_ref_usd`` / ``age_center_year`` from medians.

    Objective: weighted MSE of (empirical NM−VG+ log median gap) minus
    ``log1p_nm_equivalent`` lift for a (6,6)→(7,7) sale at the bin's median VG+ USD.
    """
    if df.empty:
        raise ValueError("sale frame is empty (check DB paths and joins)")
    df_b = add_anchor_decade_bins(df)
    bins = compute_bin_deltas(df_b, min_bin_rows=min_bin_rows, min_grade_rows=min_grade_rows)
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

    best = (float("inf"), 0.0, 0.0)
    for g in g_list:
        for ak in a_list:
            sc = grid_score_bin_medians(
                bins,
                base_alpha=base_alpha,
                base_beta=base_beta,
                price_ref_usd=price_ref,
                age_center_year=age_center,
                price_gamma=float(g),
                age_k=float(ak),
                price_scale_min=price_scale_min,
                price_scale_max=price_scale_max,
            )
            if sc < best[0]:
                best = (sc, float(g), float(ak))

    od = df["order_date"] if "order_date" in df.columns else None
    parsed_dates: list[datetime] = []
    if od is not None:
        for x in od.tolist():
            if x is None or not str(x).strip():
                continue
            d = parse_iso_datetime(str(x))
            if d is not None:
                parsed_dates.append(d)
    meta = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "fit_kind": "cross_grade_bin_median_v1",
        "nm_grade_key": nm_grade_key,
        "row_count_sales": int(len(df)),
        "bins_used": len(bins),
        "min_bin_rows": int(min_bin_rows),
        "min_grade_rows": int(min_grade_rows),
        "base_alpha": float(base_alpha),
        "base_beta": float(base_beta),
        "weighted_mse_log_lift": float(best[0]),
        "git_revision": _try_git_revision(),
    }
    if parsed_dates:
        meta["sale_date_min"] = min(parsed_dates).date().isoformat()
        meta["sale_date_max"] = max(parsed_dates).date().isoformat()
    if bins:
        meta["sale_anchor_usd_median"] = float(np.median([b.med_anchor for b in bins]))
        meta["empirical_delta_log_median_across_bins"] = float(
            np.median([b.emp_delta_log for b in bins])
        )

    return {
        "schema_version": 1,
        "fit_metadata": meta,
        "price_ref_usd": price_ref,
        "price_gamma": best[1],
        "price_scale_min": float(price_scale_min),
        "price_scale_max": float(price_scale_max),
        "age_k": best[2],
        "age_center_year": float(age_center),
    }
