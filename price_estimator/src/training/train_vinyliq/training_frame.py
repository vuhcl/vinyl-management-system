"""Load X/y training frame from marketplace + feature store (+ optional sale history)."""

from __future__ import annotations

import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...features.vinyliq_features import (
    first_artist_id,
    first_label_id,
    row_dict_for_inference,
)
from ...models.fitted_regressor import (
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    log1p_dollar_targets_for_metrics,
    mae_dollars,
    median_ape_dollars,
    pred_log1p_dollar_for_metrics,
    wape_dollars,
)
from .catalog_encoders import (
    _auto_top_k_id_encoder,
    _catalog_encoders_from_saved_bundle,
    _fit_frequency_capped_id_encoder,
)
from ..label_synthesis import training_label_config_from_vinyliq
from ..sale_floor_targets import (
    sale_floor_blend_bundle,
    sale_floor_blend_sf_cfg_for_policy,
)


def _pick_newer_marketplace_row_dict(
    a: dict[str, object],
    b: dict[str, object],
) -> dict[str, object]:
    """Prefer latest ``fetched_at`` (ISO string order); tie-break by non-null field count."""
    fa = str(a.get("fetched_at") or "")
    fb = str(b.get("fetched_at") or "")
    if fa != fb:
        return a if fa > fb else b

    def score(d: dict[str, object]) -> int:
        return sum(1 for v in d.values() if v is not None)

    return a if score(a) >= score(b) else b


def _auto_top_k_id_encoder(n_labeled: int, n_unique: int) -> int:
    if n_unique <= 0:
        return 0
    if n_unique <= 500:
        return n_unique
    k = max(500, min(3000, n_labeled // 25))
    return min(k, n_unique)


def _load_sale_history_sidecars(
    sh_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """``release_sale`` rows and ``sale_history_fetch_status`` per ``release_id``."""
    sales: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fetch: dict[str, dict[str, Any]] = {}
    if not sh_path.is_file():
        return {}, {}
    conn = sqlite3.connect(str(sh_path))
    conn.row_factory = sqlite3.Row
    try:
        for r in conn.execute("SELECT * FROM release_sale"):
            d = dict(r)
            sales[str(d["release_id"])].append(d)
        for r in conn.execute("SELECT * FROM sale_history_fetch_status"):
            d = dict(r)
            fetch[str(d["release_id"])] = d
    finally:
        conn.close()
    return dict(sales), fetch


def _default_cold_start_flags(mx: dict[str, Any]) -> dict[str, float]:
    lo = mx.get("release_lowest_price")
    try:
        has_lf = 1.0 if lo is not None and float(lo) > 0 else 0.0
    except (TypeError, ValueError):
        has_lf = 0.0
    return {"has_sale_history": 0.0, "s_imputed": 0.0, "has_listing_floor": has_lf}


@dataclass(frozen=True)
class TrainingFrameLoad:
    """Rows, targets, and dual-policy sale-history flags from ``load_training_frame``.

    ``yvals_nm`` / ``yvals_ord`` may be NaN when that policy does not produce a dollar label;
    ensemble training fits each head only on finite targets, then both heads predict on every row
    (same as production pyfunc).
    """

    xrows: list[dict]
    yvals: list[float]
    rids: list[str]
    catalog_encoders: dict[str, dict[str, float]]
    median_anchors: list[float]
    has_nm_comp_sale: list[float]
    has_ord_comp_sale: list[float]
    yvals_nm: list[float]
    yvals_ord: list[float]


def _stored_target_from_dollar_label(
    y_dollar: float | None,
    m_anchor: float | None,
    *,
    training_target_kind: str,
    residual_z_clip_abs: float | None,
) -> float:
    if y_dollar is None or m_anchor is None:
        return float("nan")
    try:
        yf = float(y_dollar)
        mf = float(m_anchor)
    except (TypeError, ValueError):
        return float("nan")
    if yf <= 0 or mf <= 0:
        return float("nan")
    if training_target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        z = float(np.log1p(yf) - np.log1p(mf))
        if residual_z_clip_abs is not None and residual_z_clip_abs > 0:
            z = float(np.clip(z, -residual_z_clip_abs, residual_z_clip_abs))
        return z
    return float(np.log1p(yf))

def load_training_frame(
    marketplace_db: Path,
    feature_store_db: Path,
    *,
    max_primary_artist_ids: int | None = None,
    max_primary_label_ids: int | None = None,
    training_label: dict[str, object] | None = None,
    training_target_kind: str = TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    residual_z_clip_abs: float | None = None,
    sale_history_db: Path | None = None,
    catalog_encoders_override: dict[str, Any] | None = None,
) -> TrainingFrameLoad:
    tl = training_label or training_label_config_from_vinyliq({})
    mode_l = str(tl.get("mode", "sale_floor_blend")).strip().lower()
    if mode_l not in ("sale_floor_blend", "sale_floor"):
        raise ValueError(
            f"Unsupported training_label.mode {mode_l!r} for VinylIQ training "
            "(expected sale_floor_blend or sale_floor)."
        )
    sales_by_rid: dict[str, list[dict[str, Any]]] = {}
    fetch_by_rid: dict[str, dict[str, Any]] = {}
    if mode_l in ("sale_floor_blend", "sale_floor"):
        if sale_history_db is None or not Path(sale_history_db).is_file():
            print(
                "Warning: sale_floor_blend needs sale_history.sqlite "
                "(vinyliq.paths.sale_history_db); sold nowcast s will be missing "
                "unless listing floor / PS-only paths yield y.",
                file=sys.stderr,
            )
        else:
            sales_by_rid, fetch_by_rid = _load_sale_history_sidecars(Path(sale_history_db))

    year_by_rid: dict[str, Any] = {}
    if Path(feature_store_db).is_file():
        conn_y = sqlite3.connect(str(feature_store_db))
        conn_y.row_factory = sqlite3.Row
        try:
            for r in conn_y.execute("SELECT release_id, year FROM releases_features"):
                year_by_rid[str(r["release_id"])] = r["year"]
        finally:
            conn_y.close()

    conn_m = sqlite3.connect(str(marketplace_db))
    conn_m.row_factory = sqlite3.Row
    cur = conn_m.execute(
        """
        SELECT release_id, fetched_at, num_for_sale,
               price_suggestions_json, release_lowest_price, release_num_for_sale,
               community_want, community_have, blocked_from_sale
        FROM marketplace_stats
        WHERE (
            release_lowest_price IS NOT NULL AND release_lowest_price > 0
        ) OR (
            price_suggestions_json IS NOT NULL
            AND TRIM(price_suggestions_json) != ''
            AND TRIM(price_suggestions_json) != '{}'
        )
        """
    )
    by_rid: dict[str, dict[str, object]] = {}
    for r in cur.fetchall():
        rid = str(r["release_id"])
        rd = {k: r[k] for k in r.keys()}
        prev = by_rid.get(rid)
        if prev is None:
            by_rid[rid] = rd
        else:
            by_rid[rid] = _pick_newer_marketplace_row_dict(prev, rd)

    labels: dict[str, float] = {}
    medians: dict[str, float] = {}
    labels_nm: dict[str, float] = {}
    medians_nm: dict[str, float] = {}
    labels_ord: dict[str, float] = {}
    medians_ord: dict[str, float] = {}
    has_nm_sale_by_rid: dict[str, float] = {}
    has_ord_sale_by_rid: dict[str, float] = {}
    marketplace_extra: dict[str, dict[str, Any]] = {}
    row_cold_flags: dict[str, dict[str, float]] = {}
    ps_grade = str(tl.get("price_suggestion_grade") or "Near Mint (NM or M-)").strip()
    sf_cfg = tl.get("sale_floor_blend") if isinstance(tl.get("sale_floor_blend"), dict) else {}
    sf_nm = sale_floor_blend_sf_cfg_for_policy(sf_cfg, "nm_substrings_only")
    sf_ord = sale_floor_blend_sf_cfg_for_policy(sf_cfg, "ordinal_cascade")
    primary_pol = str(sf_cfg.get("sale_condition_policy", "nm_substrings_only")).strip().lower()
    if primary_pol not in ("nm_substrings_only", "ordinal_cascade"):
        primary_pol = "nm_substrings_only"

    for rid, rd in by_rid.items():
        yr_raw = year_by_rid.get(rid)
        release_year: float | None
        try:
            release_year = float(yr_raw) if yr_raw is not None else None
        except (TypeError, ValueError):
            release_year = None
        if release_year is not None and not math.isfinite(release_year):
            release_year = None
        rd_d = dict(rd)
        sales = sales_by_rid.get(rid, [])
        fetch = fetch_by_rid.get(rid)
        out_nm = sale_floor_blend_bundle(
            rd_d,
            sales,
            fetch,
            sf_cfg=sf_nm,
            nm_grade_key=ps_grade,
            release_year=release_year,
        )
        out_ord = sale_floor_blend_bundle(
            rd_d,
            sales,
            fetch,
            sf_cfg=sf_ord,
            nm_grade_key=ps_grade,
            release_year=release_year,
        )
        yn, mn, fn = out_nm
        yo, mo, fo = out_ord
        has_nm_sale_by_rid[rid] = float(fn.get("has_sale_history", 0.0))
        has_ord_sale_by_rid[rid] = float(fo.get("has_sale_history", 0.0))

        if yn is not None and yn > 0:
            mnu = (
                float(mn)
                if mn is not None and float(mn) > 0
                else float(yn)
            )
            labels_nm[rid] = float(yn)
            medians_nm[rid] = mnu
        if yo is not None and yo > 0:
            mou = (
                float(mo)
                if mo is not None and float(mo) > 0
                else float(yo)
            )
            labels_ord[rid] = float(yo)
            medians_ord[rid] = mou

        y_p, m_p, flags_p = out_ord if primary_pol == "ordinal_cascade" else out_nm
        if flags_p:
            row_cold_flags[rid] = flags_p
        if y_p is not None and y_p > 0:
            m_use = (
                float(m_p)
                if m_p is not None and float(m_p) > 0
                else float(y_p)
            )
            labels[rid] = float(y_p)
            medians[rid] = m_use
            marketplace_extra[rid] = {
                "community_want": rd.get("community_want"),
                "community_have": rd.get("community_have"),
                "release_num_for_sale": rd.get("release_num_for_sale"),
                "release_lowest_price": rd.get("release_lowest_price"),
                "num_for_sale": rd.get("num_for_sale"),
                "blocked_from_sale": rd.get("blocked_from_sale"),
            }
    conn_m.close()

    conn_f = sqlite3.connect(str(feature_store_db))
    conn_f.row_factory = sqlite3.Row
    cur = conn_f.execute("SELECT * FROM releases_features")
    rows = [dict(r) for r in cur.fetchall()]
    conn_f.close()

    labeled: list[dict] = []
    for r in rows:
        rid = str(r.get("release_id", ""))
        if rid not in labels:
            continue
        y = labels[rid]
        if y <= 0:
            continue
        mx = marketplace_extra.get(rid, {})
        r_cat = dict(r)
        labeled.append(r_cat)

    artist_ids_per_row = [first_artist_id(r) for r in labeled]
    label_ids_per_row = [first_label_id(r) for r in labeled]
    n_art_u = len({x for x in artist_ids_per_row if x})
    n_lbl_u = len({x for x in label_ids_per_row if x})
    n_lab = len(labeled)

    if catalog_encoders_override is not None:
        catalog_encoders = _catalog_encoders_from_saved_bundle(catalog_encoders_override)
    else:
        if max_primary_artist_ids is not None:
            ka = max(0, int(max_primary_artist_ids))
        else:
            ka = _auto_top_k_id_encoder(n_lab, n_art_u)
        if max_primary_label_ids is not None:
            kl = max(0, int(max_primary_label_ids))
        else:
            kl = _auto_top_k_id_encoder(n_lab, n_lbl_u)

        genres_set: set[str] = set()
        countries_set: set[str] = set()
        for r in labeled:
            g = str(r.get("genre") or "").strip().lower()
            if g:
                genres_set.add(g)
            c = str(r.get("country") or "").strip().lower()
            if c:
                countries_set.add(c)

        catalog_encoders = {
            "genre": {g: float(i) for i, g in enumerate(sorted(genres_set))},
            "country": {c: float(i) for i, c in enumerate(sorted(countries_set))},
            "primary_artist_id": _fit_frequency_capped_id_encoder(artist_ids_per_row, ka),
            "primary_label_id": _fit_frequency_capped_id_encoder(label_ids_per_row, kl),
        }
        catalog_encoders["_id_encoder_meta"] = {
            "n_labeled_rows": float(n_lab),
            "primary_artist_id_unique": float(n_art_u),
            "primary_label_id_unique": float(n_lbl_u),
            "primary_artist_id_cap": float(ka),
            "primary_label_id_cap": float(kl),
        }
    g2i = catalog_encoders["genre"]
    c2i = catalog_encoders["country"]
    a2i = catalog_encoders["primary_artist_id"]
    l2i = catalog_encoders["primary_label_id"]

    inc_mkt = training_target_kind == TARGET_KIND_DOLLAR_LOG1P
    Xrows: list[dict] = []
    yvals: list[float] = []
    rids: list[str] = []
    median_anchors: list[float] = []
    has_nm_comp_sale: list[float] = []
    has_ord_comp_sale: list[float] = []
    yvals_nm: list[float] = []
    yvals_ord: list[float] = []
    for r in labeled:
        rid = str(r.get("release_id", ""))
        y_dollar = labels[rid]
        mp = medians[rid]
        gidx = g2i.get(str(r.get("genre") or "").strip().lower(), 0.0)
        cidx = c2i.get(str(r.get("country") or "").strip().lower(), 0.0)
        aidx = a2i.get(first_artist_id(r), 0.0)
        lbidx = l2i.get(first_label_id(r), 0.0)
        mx = marketplace_extra.get(rid, {})
        stats = {
            "release_lowest_price": mx.get("release_lowest_price"),
            "num_for_sale": mx.get("num_for_sale"),
            "community_want": mx.get("community_want"),
            "community_have": mx.get("community_have"),
            "release_num_for_sale": mx.get("release_num_for_sale"),
            "blocked_from_sale": mx.get("blocked_from_sale"),
        }
        cf = row_cold_flags.get(rid) or _default_cold_start_flags(mx)
        row = row_dict_for_inference(
            rid,
            "Near Mint (NM or M-)",
            "Near Mint (NM or M-)",
            stats,
            r,
            genre_index=gidx,
            country_index=cidx,
            primary_artist_index=aidx,
            primary_label_index=lbidx,
            include_marketplace_scalars_in_features=inc_mkt,
            cold_start_flags=cf,
        )
        Xrows.append(row)
        if training_target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            z = float(np.log1p(y_dollar) - np.log1p(mp))
            if residual_z_clip_abs is not None and residual_z_clip_abs > 0:
                z = float(np.clip(z, -residual_z_clip_abs, residual_z_clip_abs))
            yvals.append(z)
        else:
            yvals.append(float(np.log1p(y_dollar)))
        median_anchors.append(float(mp))
        rids.append(rid)
        has_nm_comp_sale.append(has_nm_sale_by_rid.get(rid, 0.0))
        has_ord_comp_sale.append(has_ord_sale_by_rid.get(rid, 0.0))
        y_nm = _stored_target_from_dollar_label(
            labels_nm.get(rid),
            medians_nm.get(rid),
            training_target_kind=training_target_kind,
            residual_z_clip_abs=residual_z_clip_abs,
        )
        y_ord = _stored_target_from_dollar_label(
            labels_ord.get(rid),
            medians_ord.get(rid),
            training_target_kind=training_target_kind,
            residual_z_clip_abs=residual_z_clip_abs,
        )
        yvals_nm.append(y_nm)
        yvals_ord.append(y_ord)

    return TrainingFrameLoad(
        xrows=Xrows,
        yvals=yvals,
        rids=rids,
        catalog_encoders=catalog_encoders,
        median_anchors=median_anchors,
        has_nm_comp_sale=has_nm_comp_sale,
        has_ord_comp_sale=has_ord_comp_sale,
        yvals_nm=yvals_nm,
        yvals_ord=yvals_ord,
    )


def report_residual_target_sanity(
    y_stored: np.ndarray,
    median_anchors: np.ndarray,
    target_kind: str,
    *,
    seed: int = 42,
) -> None:
    """
    Print residual ``z`` distribution, dollar metrics for constant ``z=0`` (correct anchor),
    and dollar metrics when the median anchor is shuffled (pred still ``z=0``).

    If ``|z|`` is almost always ~0, tuned models can match ~0 MdAPE without leakage.
    Shuffled-anchor MdAPE should be large unless all medians are similar or ``z`` cancels.
    """
    if target_kind != TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        return
    y = np.asarray(y_stored, dtype=np.float64)
    m = np.asarray(median_anchors, dtype=np.float64)
    az = np.abs(y)
    n = len(y)
    print("\n--- Residual target sanity ---")
    print(f"n={n}")
    print(
        f"z: mean={float(np.mean(y)):.6f}  std={float(np.std(y)):.6f}  "
        f"median|z|={float(np.median(az)):.6f}"
    )
    qs = (5, 25, 50, 75, 95, 99)
    pct = np.percentile(az, qs)
    qstr = " | ".join(f"P{p}:{float(v):.5f}" for p, v in zip(qs, pct))
    print(f"|z| percentiles: {qstr}")
    frac_tiny = float(np.mean(az < 1e-6))
    frac_small = float(np.mean(az < 0.01))
    print(
        f"frac |z|<1e-6: {100.0 * frac_tiny:.2f}%  "
        f"frac |z|<0.01: {100.0 * frac_small:.2f}%"
    )
    log1pm = np.log1p(np.maximum(m, 0.0))
    print(
        f"log1p(m_anchor): std={float(np.std(log1pm)):.5f}  "
        f"min..max={float(np.min(log1pm)):.3f}..{float(np.max(log1pm)):.3f}"
    )
    pred_z0 = np.zeros_like(y)
    y_lp = log1p_dollar_targets_for_metrics(y, m, target_kind)
    pred_lp_ok = pred_log1p_dollar_for_metrics(pred_z0, m, target_kind)
    mae0 = mae_dollars(y_lp, pred_lp_ok)
    md0 = median_ape_dollars(y_lp, pred_lp_ok)
    w0 = wape_dollars(y_lp, pred_lp_ok)
    print(
        "Constant z=0 + correct per-row anchor → "
        f"MAE ${mae0:.4f} | WAPE {100.0 * w0:.2f}% | median APE {100.0 * md0:.2f}%"
    )
    rng = np.random.default_rng(int(seed))
    m_bad = rng.permutation(m)
    pred_lp_bad = pred_log1p_dollar_for_metrics(pred_z0, m_bad, target_kind)
    mae_bad = mae_dollars(y_lp, pred_lp_bad)
    md_bad = median_ape_dollars(y_lp, pred_lp_bad)
    print(
        "Constant z=0 + shuffled anchor (destructive check) → "
        f"MAE ${mae_bad:.4f} | median APE {100.0 * md_bad:.2f}% "
        "(expect >> 0 if anchors and labels align per row)"
    )
    print("--- (end residual sanity) ---\n")
