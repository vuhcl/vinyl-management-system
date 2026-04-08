#!/usr/bin/env python3
"""
Write ``release_id`` lines from ``feature_store.sqlite`` (``releases_features``).

After a dump ingest, ``have_count`` and ``want_count`` are in the DB — use
``--sort-by have`` or ``want`` (Discogs-style popularity) without scraping.

  PYTHONPATH=. python price_estimator/scripts/export_release_ids.py \\
      --out price_estimator/data/raw/dump_release_ids.txt

  PYTHONPATH=. python price_estimator/scripts/export_release_ids.py \\
      --sort-by have --out price_estimator/data/raw/popular_by_have.txt \\
      --limit 500000

  PYTHONPATH=. python price_estimator/scripts/export_release_ids.py \\
      --sort-by want --out price_estimator/data/raw/popular_by_want.txt

Or with sqlite3 only (numeric sort only):

  sqlite3 price_estimator/data/feature_store.sqlite \\
    \"SELECT release_id FROM releases_features ORDER BY release_id\" \\
    > price_estimator/data/raw/dump_release_ids.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export release_id list from VinylIQ feature store SQLite",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "Path to feature_store.sqlite "
            "(default: price_estimator/data/feature_store.sqlite)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output text file (one release_id per line)",
    )
    parser.add_argument(
        "--sort-by",
        choices=("release_id", "have", "want"),
        default="release_id",
        help=(
            "release_id=ascending ID; have=descending have_count; "
            "want=descending want_count"
        ),
    )
    parser.add_argument(
        "--min-have",
        type=int,
        default=0,
        help="Only include releases with have_count >= N (0 = no filter)",
    )
    parser.add_argument(
        "--min-want",
        type=int,
        default=0,
        help="Only include releases with want_count >= N (0 = no filter)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max lines to write (0 = all)",
    )
    args = parser.parse_args()

    root = _root()
    db_path = args.db or (root / "data" / "feature_store.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path

    if not db_path.is_file():
        print(f"Feature store not found: {db_path}", file=sys.stderr)
        return 1

    try:
        from price_estimator.src.storage.feature_store import FeatureStoreDB
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    sort_map = {
        "release_id": "release_id",
        "have": "have_count",
        "want": "want_count",
    }
    sort_key = sort_map[args.sort_by]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    store = FeatureStoreDB(db_path)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for rid in store.iter_release_ids(
            sort_by=sort_key,
            min_have=args.min_have,
            min_want=args.min_want,
        ):
            if args.limit and n >= args.limit:
                break
            f.write(rid + "\n")
            n += 1
            if n % 500_000 == 0:
                print(f"... {n}", flush=True)
    print(f"Wrote {n} release_id lines → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
