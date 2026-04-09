#!/usr/bin/env python3
"""
Backfill ``want_count`` / ``have_count`` (and ``want_have_ratio``) on ``releases_features``
from ``GET /releases/{id}`` for rows that look dump-ingested (zero community counts).

Uses the same Discogs token as other scripts. Rate-limited sequential requests
(default ~55/min).

  PYTHONPATH=. uv run python price_estimator/scripts/backfill_feature_store_community.py \\
      --db price_estimator/data/feature_store.sqlite \\
      --limit 5000

  # Only rows with want_count+have_count == 0
  PYTHONPATH=. uv run python price_estimator/scripts/backfill_feature_store_community.py \\
      --release-ids price_estimator/data/raw/popular_release_ids.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill community want/have on feature_store from Discogs API",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="feature_store.sqlite (default: price_estimator/data/feature_store.sqlite)",
    )
    parser.add_argument(
        "--release-ids",
        type=Path,
        default=None,
        help="Optional file of release IDs to process (otherwise scan DB for zero counts)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max releases to update (0 = no limit)",
    )
    parser.add_argument(
        "--req-per-minute",
        type=float,
        default=55.0,
        help="Approximate global rate cap (default 55)",
    )
    parser.add_argument(
        "--personal-token-file",
        type=Path,
        default=None,
        help="Discogs token file (same as collect_marketplace_stats)",
    )
    args = parser.parse_args()

    try:
        from shared.project_env import load_project_dotenv

        load_project_dotenv()
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    if args.personal_token_file is not None:
        fp = args.personal_token_file.expanduser()
        if not fp.is_file():
            print(f"Token file not found: {fp}", file=sys.stderr)
            return 1
        for ln in fp.read_text(encoding="utf-8", errors="replace").splitlines():
            s = ln.replace("\ufeff", "").strip()
            if s and not s.startswith("#"):
                os.environ["DISCOGS_USER_TOKEN"] = s.strip().strip("\"'")
                break

    root = _root()
    db_path = args.db or (root / "data" / "feature_store.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path
    if not db_path.is_file():
        print(f"Feature store not found: {db_path}", file=sys.stderr)
        return 1

    from shared.discogs_api.client import discogs_client_from_env

    client = discogs_client_from_env()
    if client is None or not client.is_authenticated:
        print("Set DISCOGS_USER_TOKEN or DISCOGS_TOKEN", file=sys.stderr)
        return 1

    from price_estimator.src.storage.feature_store import FeatureStoreDB

    store = FeatureStoreDB(db_path)

    if args.release_ids is not None:
        ids: list[str] = []
        with open(args.release_ids, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip().split()[0] if line.strip() else ""
                if s.isdigit():
                    ids.append(s)
    else:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT release_id FROM releases_features "
            "WHERE COALESCE(want_count,0) + COALESCE(have_count,0) = 0"
        )
        ids = [str(r[0]) for r in cur.fetchall()]
        conn.close()

    period = 60.0 / max(1.0, float(args.req_per_minute))
    n_done = 0
    n_skip = 0
    n_err = 0
    for rid in ids:
        if args.limit and n_done >= args.limit:
            break
        row = store.get(rid)
        if row is None:
            n_skip += 1
            continue
        if args.release_ids is None:
            w0 = int(row.get("want_count") or 0)
            h0 = int(row.get("have_count") or 0)
            if w0 + h0 > 0:
                n_skip += 1
                continue
        time.sleep(period)
        try:
            rel = client.get_release_with_retries(
                rid, max_retries=6, timeout=45.0
            )
        except Exception as e:
            print(f"{rid} error: {e}", file=sys.stderr)
            n_err += 1
            continue
        if not isinstance(rel, dict):
            n_err += 1
            continue
        comm = rel.get("community") or {}
        try:
            w = int(comm.get("want") or 0)
        except (TypeError, ValueError):
            w = 0
        try:
            h = int(comm.get("have") or 0)
        except (TypeError, ValueError):
            h = 0
        ratio = (w / h) if h > 0 else 0.0
        row["want_count"] = w
        row["have_count"] = h
        row["want_have_ratio"] = ratio
        store.upsert_row(row)
        n_done += 1
        if n_done % 200 == 0:
            print(f"... updated {n_done}", flush=True)

    print(f"Done. updated={n_done} skipped={n_skip} errors={n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
