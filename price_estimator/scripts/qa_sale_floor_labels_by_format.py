"""Label QA: flag extreme sale-floor labels for 12\" / LP / box_multi buckets.

Joins the same marketplace + sale-history path as training, recomputes
``sale_floor_label_diagnostics`` per release, and prints rows sorted by how far
``y_label`` sits above listing/sold/market anchors (good for spotting bogus listings
or extrapolated nowcasts).

Usage (repo root)::

  PYTHONPATH=. uv run python price_estimator/scripts/qa_sale_floor_labels_by_format.py \\
      --top 40 --buckets twelve,lp,box_multi
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml

from price_estimator.src.features.vinyliq_features import residual_training_feature_columns
from price_estimator.src.models.fitted_regressor import mutually_exclusive_format_bucket_masks
from price_estimator.src.training.label_synthesis import training_label_config_from_vinyliq
from price_estimator.src.training.sale_floor_targets import (
    sale_floor_blend_sf_cfg_for_policy,
    sale_floor_label_diagnostics,
)
from price_estimator.src.training.train_vinyliq import (
    _load_sale_history_sidecars,
    _pick_newer_marketplace_row_dict,
    load_training_frame,
    residual_z_clip_abs_from_vinyliq,
    training_target_kind_from_vinyliq,
)


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml_config(path: Path | None) -> dict:
    if path is not None:
        p = path
    else:
        env = os.environ.get("VINYLIQ_CONFIG")
        p = Path(env) if env else _pkg_root() / "configs" / "base.yaml"
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    v = cfg.get("vinyliq") or {}
    paths = v.get("paths") or {}
    root = _pkg_root()
    d_cache = root / "data" / "cache"
    d_data = root / "data"
    mp = Path(paths.get("marketplace_db", d_cache / "marketplace_stats.sqlite"))
    fs = Path(paths.get("feature_store_db", d_data / "feature_store.sqlite"))
    sh = Path(paths.get("sale_history_db", d_cache / "sale_history.sqlite"))
    if not mp.is_absolute():
        mp = root / mp
    if not fs.is_absolute():
        fs = root / fs
    if not sh.is_absolute():
        sh = root / sh
    return mp, fs, sh


def _marketplace_by_rid(mp: Path) -> dict[str, dict[str, object]]:
    conn_m = sqlite3.connect(str(mp))
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
    conn_m.close()
    return by_rid


def _year_by_rid(fs: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    if not fs.is_file():
        return out
    conn_y = sqlite3.connect(str(fs))
    conn_y.row_factory = sqlite3.Row
    try:
        for r in conn_y.execute("SELECT release_id, year FROM releases_features"):
            out[str(r["release_id"])] = r["year"]
    finally:
        conn_y.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--buckets",
        type=str,
        default="twelve,lp,box_multi",
        help="Comma-separated: box_multi, seven, ten, twelve, lp, cd, other",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=40,
        help="How many rows to print per sort pass",
    )
    ap.add_argument(
        "--sort",
        choices=("y_over_m", "y_over_median", "lo_over_pmax"),
        default="y_over_m",
        help="Rank key: y/m_anchor, y/release_lowest_price, or raw_listing/p_max_sale",
    )
    args = ap.parse_args()

    cfg = _load_yaml_config(args.config)
    v = cfg.get("vinyliq") or {}
    mp, fs, sh = _resolve_paths(cfg)
    if not mp.is_file() or not fs.is_file():
        print("Need marketplace_stats.sqlite and feature_store.sqlite.", file=sys.stderr)
        return 1

    tl = training_label_config_from_vinyliq(v)
    target_kind = training_target_kind_from_vinyliq(v)
    z_clip = residual_z_clip_abs_from_vinyliq(v)
    ce_cfg = v.get("catalog_encoders") or {}
    max_art = int(ce_cfg["max_primary_artist_ids"]) if ce_cfg.get("max_primary_artist_ids") is not None else None
    max_lbl = int(ce_cfg["max_primary_label_ids"]) if ce_cfg.get("max_primary_label_ids") is not None else None

    sh_arg = sh if sh.is_file() else None
    frame = load_training_frame(
        mp,
        fs,
        max_primary_artist_ids=max_art,
        max_primary_label_ids=max_lbl,
        training_label=tl,
        training_target_kind=target_kind,
        residual_z_clip_abs=z_clip,
        sale_history_db=sh_arg,
    )

    cols = residual_training_feature_columns()
    X_all = np.array([[float(row[c]) for c in cols] for row in frame.xrows])
    masks = mutually_exclusive_format_bucket_masks(X_all, cols)

    want = {x.strip() for x in args.buckets.split(",") if x.strip()}
    valid = set(masks.keys())
    bad = want - valid
    if bad:
        print(f"Unknown bucket(s): {bad}. Use {sorted(valid)}", file=sys.stderr)
        return 1

    bucket_mask = np.zeros(len(frame.rids), dtype=bool)
    for name in want:
        bucket_mask |= masks[name]

    sf_cfg = tl.get("sale_floor_blend") if isinstance(tl.get("sale_floor_blend"), dict) else {}
    primary_pol = str(sf_cfg.get("sale_condition_policy", "nm_substrings_only")).strip().lower()
    if primary_pol not in ("nm_substrings_only", "ordinal_cascade"):
        primary_pol = "nm_substrings_only"
    sf_primary = sale_floor_blend_sf_cfg_for_policy(sf_cfg, primary_pol)
    ps_grade = str(tl.get("price_suggestion_grade") or "Near Mint (NM or M-)").strip()

    by_rid = _marketplace_by_rid(mp)
    year_by_rid = _year_by_rid(fs)
    sales_by_rid: dict[str, list] = {}
    fetch_by_rid: dict[str, dict] = {}
    if sh_arg is not None:
        sales_by_rid, fetch_by_rid = _load_sale_history_sidecars(sh_arg)

    # bucket name per row (priority order same as masks)
    order = ("box_multi", "seven", "ten", "twelve", "lp", "cd", "other")
    row_bucket: list[str] = []
    for i in range(len(frame.rids)):
        rb = "other"
        for name in order:
            if name in masks and bool(masks[name][i]):
                rb = name
                break
        row_bucket.append(rb)

    rows_out: list[tuple[float, str, dict[str, object]]] = []
    for i, rid in enumerate(frame.rids):
        if not bucket_mask[i]:
            continue
        rd = by_rid.get(rid)
        if rd is None:
            continue
        yr_raw = year_by_rid.get(rid)
        try:
            release_year = float(yr_raw) if yr_raw is not None else None
        except (TypeError, ValueError):
            release_year = None
        if release_year is not None and not (release_year == release_year):  # nan
            release_year = None

        rd_d = dict(rd)
        sales = sales_by_rid.get(rid, [])
        fetch = fetch_by_rid.get(rid)
        y_l, m_a, _flags, diag = sale_floor_label_diagnostics(
            rd_d,
            sales,
            fetch,
            sf_cfg=sf_primary,
            nm_grade_key=ps_grade,
            release_year=release_year,
        )
        if y_l is None or diag.get("y_label_final_usd") is None:
            continue
        y_final = float(diag["y_label_final_usd"])
        m_anchor = float(diag.get("m_anchor_usd") or m_a or 1.0)
        med_mp = diag.get("release_lowest_price_mp_usd")
        med_f = float(med_mp) if med_mp is not None else 1.0
        lo_raw = diag.get("listing_floor_raw_usd")
        pmax = diag.get("p_max_sale_observed_usd")

        if args.sort == "y_over_m":
            score = y_final / max(m_anchor, 1.0)
        elif args.sort == "y_over_median":
            score = y_final / max(med_f, 1.0)
        else:
            lo_v = float(lo_raw) if lo_raw is not None else 0.0
            pm = float(pmax) if pmax is not None else 0.0
            score = lo_v / max(pm, 1.0) if pm > 0 else lo_v

        rows_out.append(
            (
                score,
                rid,
                {
                    "bucket": row_bucket[i],
                    "y_usd": y_final,
                    "m_anchor_usd": m_anchor,
                    "median_mp_usd": med_f,
                    "listing_raw_usd": lo_raw,
                    "listing_blend_usd": diag.get("listing_floor_for_blend_usd"),
                    "sold_nowcast_usd": diag.get("sold_nowcast_usd"),
                    "p_max_sale_usd": pmax,
                    "sold_tier": diag.get("sold_tier"),
                    "n_eligible_sales": diag.get("n_eligible_sales"),
                    "relax_tag": diag.get("sale_relax_tag"),
                    "sh_ok": diag.get("sale_history_fetch_ok"),
                },
            )
        )

    rows_out.sort(key=lambda x: -x[0])
    top = rows_out[: max(1, int(args.top))]

    print(
        f"Label QA (primary policy={primary_pol!r}, sort={args.sort}, buckets={sorted(want)}, n={len(rows_out)})"
    )
    print(
        "release_id\tbucket\tscore\ty_usd\tm_anchor\tmedian_mp\tlisting_raw\t"
        "listing_blend\tsold_nowcast\tp_max_sale\ttier\tn_sales\trelax\tsh_ok"
    )
    for sc, rid, d in top:
        print(
            f"{rid}\t{d['bucket']}\t{sc:.3f}\t{d['y_usd']:.2f}\t{d['m_anchor_usd']:.2f}\t"
            f"{d['median_mp_usd']:.2f}\t{_fmt(d['listing_raw_usd'])}\t"
            f"{_fmt(d['listing_blend_usd'])}\t{_fmt(d['sold_nowcast_usd'])}\t"
            f"{_fmt(d['p_max_sale_usd'])}\t{d.get('sold_tier')}\t{d.get('n_eligible_sales')}\t"
            f"{d.get('relax_tag')}\t{d.get('sh_ok')}"
        )
    return 0


def _fmt(x: object) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return str(x)


if __name__ == "__main__":
    raise SystemExit(main())
