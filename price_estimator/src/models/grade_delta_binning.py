"""Anchor×decade bins and empirical NM−VG+ contrasts for ``grade_delta_scale`` fitting."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grade_delta_constants import (
    MISSING_YEAR_DECADE_SENTINEL,
    ORDINAL_NM,
    ORDINAL_VG_PLUS_HI,
    ORDINAL_VG_PLUS_LO,
)


def add_anchor_decade_bins(
    df: pd.DataFrame,
    *,
    n_anchor_bins: int = 10,
) -> pd.DataFrame:
    """Adds ``anchor_decile`` (0..n-1) and ``decade`` (floor year / 10 * 10; NaN → sentinel)."""
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
    decade = np.where(
        np.isfinite(yv),
        np.floor(yv / 10.0) * 10.0,
        MISSING_YEAR_DECADE_SENTINEL,
    )
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
    nm = df["eff_ord"] >= ORDINAL_NM
    # Strict VG+: both media and sleeve in [6, 7), excluding mixed NM sleeve/media rows.
    vgp = (
        (df["media_ord"] >= ORDINAL_VG_PLUS_LO)
        & (df["media_ord"] < ORDINAL_VG_PLUS_HI)
        & (df["sleeve_ord"] >= ORDINAL_VG_PLUS_LO)
        & (df["sleeve_ord"] < ORDINAL_VG_PLUS_HI)
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
    nm_m = df["eff_ord"] >= ORDINAL_NM
    vgp_sym_m = (
        (df["media_ord"] >= ORDINAL_VG_PLUS_LO)
        & (df["media_ord"] < ORDINAL_VG_PLUS_HI)
        & (df["sleeve_ord"] >= ORDINAL_VG_PLUS_LO)
        & (df["sleeve_ord"] < ORDINAL_VG_PLUS_HI)
    )
    media_slice_m = (
        (df["media_ord"] >= ORDINAL_VG_PLUS_LO)
        & (df["media_ord"] < ORDINAL_VG_PLUS_HI)
        & (df["sleeve_ord"] >= ORDINAL_NM)
    )
    sleeve_slice_m = (
        (df["media_ord"] >= ORDINAL_NM)
        & (df["sleeve_ord"] >= ORDINAL_VG_PLUS_LO)
        & (df["sleeve_ord"] < ORDINAL_VG_PLUS_HI)
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
