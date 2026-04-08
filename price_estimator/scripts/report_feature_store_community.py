#!/usr/bin/env python3
"""
Summarize have_count / want_count in ``releases_features`` (feature store).

Use this to confirm whether ``build_stats_collection_queue.py`` with
``--rank-by combined`` (community sort) can rank by demand. If every row is
zero, use ``--rank-by proxy`` instead. Dump-only stores typically have all
zeros for have/want.

  uv run python price_estimator/scripts/report_feature_store_community.py \\
      --db price_estimator/data/feature_store.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "feature_store.sqlite "
            "(default: price_estimator/data/feature_store.sqlite)"
        ),
    )
    args = p.parse_args()
    root = _root()
    db_path = args.db or (root / "data" / "feature_store.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path
    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM releases_features").fetchone()[0]
        nz = conn.execute(
            "SELECT COUNT(*) FROM releases_features "
            "WHERE COALESCE(have_count,0) + COALESCE(want_count,0) > 0"
        ).fetchone()[0]
        mx = conn.execute(
            "SELECT MAX(COALESCE(have_count,0) + COALESCE(want_count,0)) "
            "FROM releases_features"
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"feature store: {db_path}")
    print(f"  total rows: {total}")
    print(f"  rows with have+want > 0: {nz}")
    print(f"  max have+want: {mx}")
    if nz == 0:
        print(
            "\nAll community counts are zero — popularity-based queue order "
            "is equivalent to release_id sort. "
            "See ingest_discogs_dump --probe-community.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
