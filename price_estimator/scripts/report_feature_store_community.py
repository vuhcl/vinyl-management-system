#!/usr/bin/env python3
"""
Summarize ``community_want`` + ``community_have`` in ``marketplace_stats.sqlite``.

Plan §1b: the feature store no longer stores community columns; use this script
to see whether popularity-based queue ordering has signal.
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
            "marketplace_stats.sqlite "
            "(default: price_estimator/data/cache/marketplace_stats.sqlite)"
        ),
    )
    args = p.parse_args()
    root = _root()
    db_path = args.db or (root / "data" / "cache" / "marketplace_stats.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path
    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM marketplace_stats").fetchone()[0]
        nz = conn.execute(
            "SELECT COUNT(*) FROM marketplace_stats "
            "WHERE COALESCE(community_have,0) + COALESCE(community_want,0) > 0"
        ).fetchone()[0]
        mx = conn.execute(
            "SELECT MAX(COALESCE(community_have,0) + COALESCE(community_want,0)) "
            "FROM marketplace_stats"
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"marketplace_stats: {db_path}")
    print(f"  total rows: {total}")
    print(f"  rows with have+want > 0: {nz}")
    print(f"  max have+want: {mx}")
    if nz == 0:
        print(
            "\nAll community counts are zero — use catalog proxy for queue order.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
