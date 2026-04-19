#!/usr/bin/env python3
"""
Build releases_features SQLite from a CSV (extract from Discogs dump or manual seed).

Expected CSV columns (header row):
  release_id, genre, year, country, format_desc, master_id, master_year
  Optional: formats_json (Discogs-style list with ``descriptions`` for §1a Repress rule).

Community counts are **not** stored (plan §1b); use ``marketplace_stats`` at training time.

Usage:
  PYTHONPATH=. python price_estimator/scripts/build_feature_store.py \\
      --input price_estimator/data/raw/releases_features_sample.csv \\
      --db price_estimator/data/feature_store.sqlite
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def row_from_csv(rec: dict[str, str]) -> dict:
    def i(x: str, d: int = 0) -> int:
        try:
            return int(float(x)) if x.strip() else d
        except (ValueError, TypeError):
            return d

    def f(x: str, d: float = 0.0) -> float:
        try:
            return float(x) if x.strip() else d
        except (ValueError, TypeError):
            return d

    rid = rec.get("release_id", "").strip()
    year = i(rec.get("year", "0"))
    decade = (year // 10) * 10 if year else 0
    fmt = rec.get("format_desc", "") or ""
    from price_estimator.src.features.vinyliq_features import (
        format_flags_from_text,
        is_original_pressing_from_format_desc,
        is_original_pressing_from_formats_json,
    )

    flags = format_flags_from_text(fmt)
    fj = (rec.get("formats_json") or "").strip()
    if fj:
        is_orig = is_original_pressing_from_formats_json(fj)
    else:
        is_orig = is_original_pressing_from_format_desc(fmt)
    row = {
        "release_id": rid,
        "master_id": rec.get("master_id", "").strip() or None,
        "genre": rec.get("genre", "").strip() or None,
        "style": rec.get("style", "").strip() or None,
        "decade": decade,
        "year": year,
        "country": rec.get("country", "").strip() or None,
        "label_tier": i(rec.get("label_tier", "0")),
        "is_original_pressing": is_orig,
        "is_colored_vinyl": flags["is_colored_vinyl"],
        "is_picture_disc": flags["is_picture_disc"],
        "is_promo": flags["is_promo"],
        "format_desc": fmt or None,
        "artists_json": (rec.get("artists_json") or "").strip() or None,
        "labels_json": (rec.get("labels_json") or "").strip() or None,
        "genres_json": (rec.get("genres_json") or "").strip() or None,
        "styles_json": (rec.get("styles_json") or "").strip() or None,
        "formats_json": (rec.get("formats_json") or "").strip() or None,
    }
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--db", type=Path, default=None)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    db_path = args.db or (root / "data" / "feature_store.sqlite")

    from price_estimator.src.storage.feature_store import FeatureStoreDB

    store = FeatureStoreDB(db_path)
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n = 0
        for rec in reader:
            row = row_from_csv(rec)
            if not row["release_id"]:
                continue
            store.upsert_row(row)
            n += 1
    print(f"Upserted {n} rows into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
