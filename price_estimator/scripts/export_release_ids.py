#!/usr/bin/env python3
"""
Write ``release_id`` lines from ``feature_store.sqlite`` (``releases_features``).

Community sorts use ``marketplace_stats.community_have`` / ``community_want``
(plan §1b), not the feature store. Pass ``--marketplace-db`` for have/want/combined.
Use ``--sort-by catalog_proxy`` for catalog-only ordering.

  PYTHONPATH=. python price_estimator/scripts/export_release_ids.py \\
      --out price_estimator/data/raw/dump_release_ids.txt

  PYTHONPATH=. python price_estimator/scripts/export_release_ids.py \\
      --sort-by have --out price_estimator/data/raw/popular_by_have.txt \\
      --limit 500000

  PYTHONPATH=. python price_estimator/scripts/export_release_ids.py \\
      --sort-by want --out price_estimator/data/raw/popular_by_want.txt

  PYTHONPATH=. python price_estimator/scripts/export_release_ids.py \\
      --sort-by combined --out price_estimator/data/raw/popular_by_have_plus_want.txt

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
        "--marketplace-db",
        type=Path,
        default=None,
        help=(
            "marketplace_stats.sqlite (required for --sort-by have|want|combined; "
            "default: <parent of --db>/cache/marketplace_stats.sqlite)"
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
        choices=("release_id", "have", "want", "combined", "catalog_proxy"),
        default="release_id",
        help=(
            "release_id=ascending ID; have/want/combined=community sorts; "
            "catalog_proxy=master+artist catalog mass (JSON1)"
        ),
    )
    parser.add_argument(
        "--proxy-weight-master",
        type=float,
        default=1.0,
        help="catalog_proxy only: weight per master fan-out (default 1)",
    )
    parser.add_argument(
        "--proxy-weight-artist",
        type=float,
        default=1.0,
        help="catalog_proxy only: weight per primary artist catalog mass (default 1)",
    )
    parser.add_argument(
        "--min-have",
        type=int,
        default=0,
        help="Only include releases with community_have >= N (0 = no filter)",
    )
    parser.add_argument(
        "--min-want",
        type=int,
        default=0,
        help="Only include releases with community_want >= N (0 = no filter)",
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

    mp_path = args.marketplace_db or (db_path.parent / "cache" / "marketplace_stats.sqlite")
    if not mp_path.is_absolute():
        mp_path = root / mp_path
    if args.sort_by in ("have", "want", "combined") and not mp_path.is_file():
        print(
            f"marketplace_stats.sqlite not found ({mp_path}); "
            "required for community sort.",
            file=sys.stderr,
        )
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
        "combined": "popularity",
        "catalog_proxy": "catalog_proxy",
    }
    sort_key = sort_map[args.sort_by]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    store = FeatureStoreDB(db_path)
    mp_kw = (
        {"marketplace_db_path": str(mp_path.resolve())}
        if args.sort_by in ("have", "want", "combined")
        else {}
    )
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for rid in store.iter_release_ids(
            sort_by=sort_key,
            min_have=args.min_have,
            min_want=args.min_want,
            proxy_weight_master=args.proxy_weight_master,
            proxy_weight_artist=args.proxy_weight_artist,
            **mp_kw,
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
