#!/usr/bin/env python3
"""
Ingest real VinylIQ training data from the Discogs API.

For each release ID:
  - GET /releases/{id} → feature store (want/have, genre, year, formats, …)
  - GET /marketplace/stats/{id} → labels / cache (median price, etc.)

Release IDs can come from a text file (one ID per line) or from a user's
collection + wantlist (needs DISCOGS_USER_TOKEN or DISCOGS_TOKEN; username must match the
token for private folders).

Usage (repo root, token set):

  PYTHONPATH=. python price_estimator/scripts/ingest_from_discogs.py \\
      --release-ids price_estimator/data/raw/my_release_ids.txt

  PYTHONPATH=. python price_estimator/scripts/ingest_from_discogs.py \\
      --username YourDiscogsUser

  # Features only (no marketplace calls):
  PYTHONPATH=. python price_estimator/scripts/ingest_from_discogs.py \\
      --release-ids ids.txt --no-stats

  # Stats only:
  PYTHONPATH=. python price_estimator/scripts/ingest_from_discogs.py \\
      --release-ids ids.txt --no-features

Then train:

  PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ids_from_file(path: Path) -> list[str]:
    lines = [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    out: list[str] = []
    for ln in lines:
        # allow "12345 # comment"
        token = ln.split()[0] if ln.split() else ""
        if token.isdigit():
            out.append(token)
    return out


def collect_ids_from_user(
    client: Any,
    username: str,
    *,
    wantlist: bool,
    collection: bool,
) -> list[str]:
    ids: set[str] = set()
    if collection:
        for r in client.get_user_collection_releases(username, folder_id=0):
            rid = r.get("id") or (r.get("basic_information") or {}).get("id")
            if rid is not None:
                ids.add(str(rid))
    if wantlist:
        for w in client.get_user_wantlist(username):
            if w.get("id") is not None:
                ids.add(str(w["id"]))
    return sorted(ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Discogs release + marketplace data for VinylIQ",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--release-ids",
        type=Path,
        help="Text file: one release ID per line (# comments ok)",
    )
    src.add_argument(
        "--username",
        type=str,
        help="Discogs username: union of collection (folder 0) + wantlist",
    )
    parser.add_argument(
        "--feature-db",
        type=Path,
        default=None,
        help=(
            "SQLite for releases_features "
            "(default: price_estimator/data/feature_store.sqlite)"
        ),
    )
    parser.add_argument(
        "--marketplace-db",
        type=Path,
        default=None,
        help=(
            "SQLite for marketplace_stats "
            "(default: price_estimator/data/cache/marketplace_stats.sqlite)"
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between releases (rate limit)",
    )
    parser.add_argument("--jitter", type=float, default=0.4)
    parser.add_argument(
        "--fetch-master",
        action="store_true",
        help="Extra GET /masters/{id} per release (is_original_pressing)",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip GET /releases (only marketplace stats)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip GET /marketplace/stats (only features)",
    )
    args = parser.parse_args()

    if args.no_features and args.no_stats:
        print("Cannot use both --no-features and --no-stats", file=sys.stderr)
        return 1

    try:
        from shared.project_env import load_project_dotenv

        load_project_dotenv()
    except ImportError:
        pass

    try:
        from shared.discogs_api.client import discogs_client_from_env
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    client = discogs_client_from_env()
    if client is None or not client.is_authenticated:
        print(
            "No Discogs credentials: set DISCOGS_TOKEN in .env or OAuth "
            "(see collect_marketplace_stats.py --help).",
            file=sys.stderr,
        )
        return 1

    from price_estimator.src.ingest.release_parser import release_to_feature_row
    from price_estimator.src.storage.feature_store import FeatureStoreDB
    from price_estimator.src.storage.marketplace_db import MarketplaceStatsDB

    root = _root()
    feat_path = args.feature_db or (root / "data" / "feature_store.sqlite")
    default_mkt = root / "data" / "cache" / "marketplace_stats.sqlite"
    mkt_path = args.marketplace_db or default_mkt
    if not feat_path.is_absolute():
        feat_path = root / feat_path
    if not mkt_path.is_absolute():
        mkt_path = root / mkt_path

    features = FeatureStoreDB(feat_path)
    market = MarketplaceStatsDB(mkt_path)

    if args.release_ids:
        ids = load_ids_from_file(args.release_ids)
    else:
        ids = collect_ids_from_user(
            client,
            args.username.strip(),
            wantlist=True,
            collection=True,
        )

    ids = sorted(set(ids))
    if not ids:
        print("No release IDs to ingest.", file=sys.stderr)
        return 1

    feat_on = not args.no_features
    stats_on = not args.no_stats
    print(f"Ingesting {len(ids)} releases (features={feat_on}, stats={stats_on})")

    n = len(ids)
    for i, rid in enumerate(ids):
        try:
            if not args.no_features:
                rel = client.get_release(rid)
                if not isinstance(rel, dict):
                    print(
                        f"[{i + 1}/{n}] {rid} release: bad response",
                        file=sys.stderr,
                    )
                else:
                    master_json = None
                    if args.fetch_master and rel.get("master_id"):
                        try:
                            master_json = client.get_master(rel["master_id"])
                        except Exception as me:
                            print(
                                f"[{i + 1}/{n}] {rid} master warn: {me}",
                                file=sys.stderr,
                            )
                    row = release_to_feature_row(rel, master=master_json)
                    features.upsert_row(row)
                    print(f"[{i + 1}/{n}] {rid} features ok")
            if not args.no_stats:
                raw = client.get_marketplace_stats(rid)
                if not isinstance(raw, dict):
                    raw = {}
                market.upsert(rid, raw)
                print(f"[{i + 1}/{n}] {rid} stats ok")
        except Exception as e:
            print(f"[{i + 1}/{n}] {rid} ERROR: {e}", file=sys.stderr)

        time.sleep(args.delay + random.uniform(0, args.jitter))

    print(f"Done. feature_store={feat_path} marketplace={mkt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
