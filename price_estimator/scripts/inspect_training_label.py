#!/usr/bin/env python3
"""Training label inspection for a single release_id (sale-floor blend diagnostics)."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import yaml

from price_estimator.src.storage.marketplace_db import price_suggestion_values_by_grade
from price_estimator.src.training.label_synthesis import training_label_config_from_vinyliq
from price_estimator.src.training.sale_floor_inference import max_price_suggestion_ladder_usd
from price_estimator.src.training.sale_floor_targets import (
    inference_residual_anchor_usd,
    sale_floor_blend_sf_cfg_for_policy,
    sale_floor_label_diagnostics,
)
from price_estimator.src.training.train_vinyliq.training_frame import _load_sale_history_sidecars


def _row_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("release_id", type=str)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("price_estimator/configs/base.yaml"),
    )
    args = ap.parse_args()
    rel = str(args.release_id).strip()
    root = Path("price_estimator")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["vinyliq"]
    tl = training_label_config_from_vinyliq(cfg)
    sf = tl.get("sale_floor_blend") or {}
    pol = str(sf.get("sale_condition_policy", "nm_substrings_only"))
    sf_cfg = sale_floor_blend_sf_cfg_for_policy(sf, pol)
    ps_grade = str(tl.get("price_suggestion_grade") or "Near Mint (NM or M-)")

    paths = cfg.get("paths") or {}
    mp = root / str(paths.get("marketplace_db", "data/cache/marketplace_stats.sqlite"))
    fs = root / str(paths.get("feature_store_db", "data/feature_store.sqlite"))
    sh = root / str(paths.get("sale_history_db", "data/cache/sale_history.sqlite"))

    if not mp.is_file():
        print(f"Missing marketplace db: {mp}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(mp)
    conn.row_factory = sqlite3.Row
    mp_row = _row_dict(
        conn.execute(
            "SELECT * FROM marketplace_stats WHERE release_id=?", (rel,)
        ).fetchone()
    )
    conn.close()
    if not mp_row:
        print(f"No marketplace_stats row for {rel}", file=sys.stderr)
        return 1

    fs_row = None
    yr = None
    if fs.is_file():
        conn = sqlite3.connect(fs)
        conn.row_factory = sqlite3.Row
        fs_row = _row_dict(
            conn.execute(
                "SELECT release_id, year, genre, country FROM releases_features WHERE release_id=?",
                (rel,),
            ).fetchone()
        )
        conn.close()
        if fs_row and fs_row.get("year") is not None:
            yr = float(fs_row["year"])

    sales: dict = {}
    fetch: dict = {}
    n_sales = 0
    if sh.is_file():
        sales, fetch = _load_sale_history_sidecars(sh)
        n_sales = len(sales.get(rel, []))

    y, m, flags, diag = sale_floor_label_diagnostics(
        mp_row,
        sales.get(rel, []),
        fetch.get(rel),
        sf_cfg=sf_cfg,
        nm_grade_key=ps_grade,
        release_year=yr,
    )

    z = math.log1p(y) - math.log1p(m) if y and m and y > 0 and m > 0 else None
    ladder = price_suggestion_values_by_grade(mp_row.get("price_suggestions_json"))

    out = {
        "release_id": rel,
        "feature_store": fs_row,
        "in_training_frame": fs_row is not None and y is not None and y > 0,
        "sale_history_rows": n_sales,
        "fetch_status": fetch.get(rel),
        "y_label_usd": y,
        "m_anchor_usd_training": m,
        "residual_z_training": z,
        "reconstructed_y_from_z": (
            math.expm1(z + math.log1p(m)) if z is not None else None
        ),
        "inference_residual_anchor_usd": inference_residual_anchor_usd(
            mp_row, nm_grade_key=ps_grade
        ),
        "ps_ladder_max_usd": max_price_suggestion_ladder_usd(mp_row),
        "ps_ladder_rungs": ladder,
        "release_lowest_price": mp_row.get("release_lowest_price"),
        "num_for_sale": mp_row.get("num_for_sale"),
        "flags": flags,
        "diagnostics": diag,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
