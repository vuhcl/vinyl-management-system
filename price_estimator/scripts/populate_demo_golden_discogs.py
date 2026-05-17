#!/usr/bin/env python3
"""
Populate local SQLite stores for the demo golden release ID.

Reads ``demo_release_id`` from ``grader/demo/golden_predict_demo.json`` (or another
JSON with the same top-level keys) and:

- ``GET /releases/{id}`` → ``feature_store`` + ``marketplace_stats`` listing fields
- ``GET /marketplace/price_suggestions/{id}`` → ``price_suggestions_json``

This matches ``collect_marketplace_stats.py --collect-mode full`` (two API calls / id).

Requires Discogs credentials (same as collectors). Repo root must be on
``PYTHONPATH`` for ``shared.discogs_api``::

  PYTHONPATH=. uv run python price_estimator/scripts/populate_demo_golden_discogs.py

Then bulk-load Postgres for GKE::

  PYTHONPATH=. uv run python price_estimator/scripts/sqlite_to_cloudsql_loader.py \\
    --feature-store price_estimator/data/feature_store.sqlite \\
    --marketplace-db price_estimator/data/cache/marketplace_stats.sqlite \\
    --database-url \"$DATABASE_URL\"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _pkg_root().parent


def _load_golden(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("golden JSON must be an object")
    rid = raw.get("demo_release_id")
    if rid is None:
        raise ValueError("golden JSON must include demo_release_id")
    sid = str(rid).strip()
    if not sid.isdigit():
        raise ValueError("golden JSON demo_release_id must be a numeric Discogs release id")
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Discogs hydrate for demo_release_id from golden_predict_demo.json",
    )
    ap.add_argument(
        "--golden-file",
        type=Path,
        default=None,
        help="JSON with demo_release_id (default: grader/demo/golden_predict_demo.json)",
    )
    ap.add_argument(
        "--feature-db",
        type=Path,
        default=None,
        help="releases_features SQLite (default: price_estimator/data/feature_store.sqlite)",
    )
    ap.add_argument(
        "--marketplace-db",
        type=Path,
        default=None,
        help="marketplace_stats SQLite "
        "(default: price_estimator/data/cache/marketplace_stats.sqlite)",
    )
    ap.add_argument("--fetch-master", action="store_true")
    ap.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds after each release API pair (rate limit hygiene)",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(_repo_root()))

    try:
        from shared.project_env import load_project_dotenv

        load_project_dotenv()
    except ImportError:
        pass

    from shared.discogs_api.client import discogs_client_from_env

    client = discogs_client_from_env()
    if client is None or not client.is_authenticated:
        print(
            "No Discogs credentials: set DISCOGS_TOKEN / OAuth "
            "(see collect_marketplace_stats.py).",
            file=sys.stderr,
        )
        return 1

    pkg = _pkg_root()
    gf = (
        Path(args.golden_file)
        if args.golden_file is not None
        else (_repo_root() / "grader" / "demo" / "golden_predict_demo.json")
    )
    if not gf.is_file():
        print(f"Golden file not found: {gf}", file=sys.stderr)
        return 1

    try:
        golden = _load_golden(gf)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        print(f"golden file: {e}", file=sys.stderr)
        return 1

    rid = str(golden["demo_release_id"]).strip()

    feat_path = args.feature_db or (pkg / "data" / "feature_store.sqlite")
    mkt_path = args.marketplace_db or (
        pkg / "data" / "cache" / "marketplace_stats.sqlite"
    )
    if not feat_path.is_absolute():
        feat_path = pkg / feat_path
    if not mkt_path.is_absolute():
        mkt_path = pkg / mkt_path

    from price_estimator.src.ingest.release_parser import release_to_feature_row
    from price_estimator.src.storage.feature_store import FeatureStoreDB
    from price_estimator.src.storage.marketplace_db import MarketplaceStatsDB

    features = FeatureStoreDB(feat_path)
    market = MarketplaceStatsDB(mkt_path)

    release_pl = client.get_release_with_retries(
        rid,
        max_retries=4,
        backoff_base=1.5,
        backoff_max=120.0,
        timeout=45.0,
    )
    time.sleep(max(0.0, args.delay) + random.uniform(0.0, 0.35))

    raw_ps = client.get_price_suggestions_with_retries(
        rid,
        max_retries=4,
        backoff_base=1.5,
        backoff_max=120.0,
        timeout=45.0,
    )
    ps_pl: dict[str, Any] = raw_ps if isinstance(raw_ps, dict) else {}

    rel = release_pl if isinstance(release_pl, dict) else {}

    master_json = None
    if args.fetch_master and rel.get("master_id"):
        try:
            master_json = client.get_master(rel["master_id"])
        except Exception as me:
            print(f"{rid} GET /masters warn: {me}", file=sys.stderr)

    if str(rel.get("id", "")).strip() == rid:
        row = release_to_feature_row(rel, master=master_json)
        features.upsert_row(row)
        print(f"{rid}: features OK -> {feat_path}")
    else:
        print(
            f"{rid}: SKIP features (missing or mismatched GET /releases payload)",
            file=sys.stderr,
        )

    release_payload = rel if str(rel.get("id", "")).strip() == rid else None
    market.upsert(
        rid,
        {},
        release_payload=release_payload,
        price_suggestions_payload=ps_pl,
    )
    print(f"{rid}: marketplace full-mode OK ({len(ps_pl)} PS keys) -> {mkt_path}")
    print("Next: sqlite_to_cloudsql_loader.py for Cloud SQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
