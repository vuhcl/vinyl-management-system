#!/usr/bin/env python3
"""
Phase 2a: build ``discogs_master_to_aoty.json`` from Discogs dump tables.

Joins ``masters_features`` (ingested masters XML) to ``recommender/data/processed/albums.parquet``
on normalized artist + title + year (±1), restricted to masters that have at
least one **album-ish** row in ``releases_features`` (``Album`` / ``LP`` in
``format_desc`` or format descriptions).

Merge with an existing JSON map by default (new ``master_id`` keys only;
never drops API-derived rows). Prints coverage vs ``albums.parquet``.

Example::

  PYTHONPATH=. uv run python scripts/build_discogs_master_to_aoty_dump_join.py \\
      --albums recommender/data/processed/albums.parquet \\
      --feature-db price_estimator/data/feature_store.sqlite \\
      --output artifacts/discogs_master_to_aoty.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2a: Discogs masters dump ↔ AOTY albums join "
            "(album-ish masters only)."
        )
    )
    parser.add_argument(
        "--albums",
        type=Path,
        default=Path("recommender/data/processed/albums.parquet"),
        help="AOTY albums parquet (album_id, artist, album_title, year)",
    )
    parser.add_argument(
        "--feature-db",
        type=Path,
        default=Path("price_estimator/data/feature_store.sqlite"),
        help="SQLite with masters_features + releases_features",
    )
    parser.add_argument(
        "--existing-json",
        type=Path,
        default=Path("artifacts/discogs_master_to_aoty.json"),
        help="Optional existing master→AOTY map to merge into (may be missing).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/discogs_master_to_aoty.json"),
    )
    parser.add_argument(
        "--no-merge-existing",
        action="store_true",
        help="Ignore --existing-json; write only dump-join discoveries.",
    )
    parser.add_argument(
        "--min-fuzzy-title-similarity",
        type=float,
        default=0.88,
        help="Minimum fuzzy title match when no exact normalized-title hit.",
    )
    parser.add_argument(
        "--year-window",
        type=int,
        default=1,
        help="Match when |master_year - album_year| <= this (0 = exact year only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only; do not write JSON.",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
        from recommender.src.data.discogs_aoty_id_matching import (
            load_master_to_aoty_json,
            save_master_to_aoty_json,
        )
        from recommender.src.retrieval.dump_master_aoty_join import (
            DumpJoinConfig,
            dump_join_stats,
            fetch_albumish_master_ids,
            load_masters_for_join,
            match_albums_dump_join,
            merge_master_to_aoty_maps,
        )
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    albums_path = args.albums
    if not albums_path.is_absolute():
        albums_path = _REPO_ROOT / albums_path
    feat_db = args.feature_db
    if not feat_db.is_absolute():
        feat_db = _REPO_ROOT / feat_db
    out_path = args.output
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    existing_path = args.existing_json
    if not existing_path.is_absolute():
        existing_path = _REPO_ROOT / existing_path

    if not albums_path.is_file():
        print(f"Albums parquet not found: {albums_path}", file=sys.stderr)
        return 1
    if not feat_db.is_file():
        print(f"Feature store not found: {feat_db}", file=sys.stderr)
        return 1

    albums = pd.read_parquet(albums_path)
    conn = sqlite3.connect(str(feat_db))
    try:
        eligible = fetch_albumish_master_ids(conn)
    finally:
        conn.close()

    print(f"album-ish master_ids (distinct): {len(eligible):,}")
    if not eligible:
        print("No album-ish masters found; check releases_features content.")
        return 2

    masters = load_masters_for_join(feat_db, eligible)
    print(f"masters_features rows (eligible): {len(masters):,}")
    if masters.empty:
        print("masters_features join returned no rows.", file=sys.stderr)
        return 3

    cfg = DumpJoinConfig(
        min_fuzzy_title_similarity=float(args.min_fuzzy_title_similarity),
        year_window=max(0, int(args.year_window)),
    )
    discovered = match_albums_dump_join(albums, masters, cfg)
    print(f"dump-join discoveries (master→album): {len(discovered):,}")

    existing: dict[str, str] = {}
    if not args.no_merge_existing and existing_path.is_file():
        existing = load_master_to_aoty_json(existing_path)
        print(f"existing map entries: {len(existing):,}")

    merged = merge_master_to_aoty_maps(existing, discovered)
    stats = dump_join_stats(albums, merged)
    print(
        "coverage vs albums.parquet: "
        f"{stats['n_albums_mapped']}/{stats['n_albums']} "
        f"({100.0 * float(stats['coverage']):.2f}%)"
    )
    print(json.dumps({k: stats[k] for k in stats if k != "coverage"}, indent=2))
    print(f"coverage_ratio: {stats['coverage']:.6f}")

    if args.dry_run:
        print("dry-run: not writing JSON")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_master_to_aoty_json(out_path, merged)
    print(f"Wrote {len(merged):,} entries → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
