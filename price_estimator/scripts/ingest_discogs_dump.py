#!/usr/bin/env python3
"""
Build ``releases_features`` SQLite from a Discogs monthly **releases** XML dump.

Download ``discogs_*_releases.xml.gz`` from https://data.discogs.com/ (CC0), then:

  PYTHONPATH=. python price_estimator/scripts/ingest_discogs_dump.py \\
      --dump /path/to/discogs_20240201_releases.xml.gz

Optional: write every release_id to a text file for
``collect_marketplace_stats.py`` (price labels):

  PYTHONPATH=. python price_estimator/scripts/ingest_discogs_dump.py \\
      --dump releases.xml.gz \\
      --ids-out price_estimator/data/raw/dump_release_ids.txt

**Community have/want:** Many monthly ``releases.xml.gz`` files do **not** include
``<community>`` want/have counts (or they are empty). Then every row ingests as
``0/0`` and ``--rank-by combined`` / ``have`` / ``want`` sort only by
``release_id``. Default ``--rank-by proxy`` uses catalog-based ranking.
Use ``--probe-community 5000`` on your dump to verify before
a full ingest; fill counts via Discogs **API** (e.g. ``ingest_from_discogs.py``)
if you need popularity ordering.

Use ``--limit`` for a smoke test; ``--ids-only`` to only emit IDs (no SQLite).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Discogs releases XML dump into VinylIQ feature store",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        required=True,
        help="Path to releases.xml or releases.xml.gz",
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
        "--ids-out",
        type=Path,
        default=None,
        help="Append one release_id per line (created or overwritten)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many releases (0 = no limit)",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=500,
        help="Rows per SQLite transaction when writing features",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Only write --ids-out; do not touch feature store",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only; print counts, no writes",
    )
    parser.add_argument(
        "--probe-community",
        type=int,
        metavar="N",
        default=0,
        help=(
            "Parse first N releases, print how many have non-zero have/want in XML; "
            "then exit (no DB write). Use to verify dump has community stats."
        ),
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Keep releases with status=Deleted (default: skip)",
    )
    args = parser.parse_args()

    dump_path = args.dump
    if not dump_path.is_file():
        print(f"Dump file not found: {dump_path}", file=sys.stderr)
        return 1

    root = _root()
    skip_deleted = not args.include_deleted

    try:
        from price_estimator.src.ingest.discogs_dump import (
            iter_dump_feature_rows,
            probe_dump_community,
        )
        from price_estimator.src.storage.feature_store import FeatureStoreDB
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    if args.probe_community > 0:
        parsed, nz, mx = probe_dump_community(
            dump_path,
            limit=int(args.probe_community),
            skip_deleted=skip_deleted,
        )
        print(
            f"probe-community: scanned {parsed} releases "
            f"(limit was {args.probe_community})"
        )
        print(f"  releases with have+want > 0 in XML: {nz}")
        print(f"  max have+want seen: {mx}")
        if nz == 0:
            print(
                "\nNo community stats in this sample — dump-ingested DB will have "
                "all zeros; popularity queues sort by release_id only.\n"
                "Community counts live in marketplace_stats (collect_marketplace_stats).",
                file=sys.stderr,
            )
        return 0

    feat_path = args.feature_db or (root / "data" / "feature_store.sqlite")
    if not feat_path.is_absolute():
        feat_path = root / feat_path

    ids_f = None
    if args.ids_out and not args.dry_run:
        args.ids_out.parent.mkdir(parents=True, exist_ok=True)
        ids_f = open(args.ids_out, "w", encoding="utf-8")

    db: FeatureStoreDB | None = None
    if not args.ids_only and not args.dry_run:
        db = FeatureStoreDB(feat_path)

    rows_iter = iter_dump_feature_rows(dump_path, skip_deleted=skip_deleted)
    batch: list[dict] = []
    total = 0
    try:
        for row in rows_iter:
            if args.limit and total >= args.limit:
                break
            total += 1
            if ids_f:
                ids_f.write(row["release_id"] + "\n")
            if args.dry_run:
                continue
            if args.ids_only:
                continue
            batch.append(row)
            if len(batch) >= max(1, args.commit_every):
                db.upsert_many(batch)
                batch.clear()
                print(f"... {total} releases", flush=True)
        if batch and db is not None:
            db.upsert_many(batch)
    finally:
        if ids_f:
            ids_f.close()

    if args.dry_run:
        print(f"dry-run: parsed {total} releases (no writes)")
    else:
        msg = f"done: {total} releases"
        if db is not None:
            msg += f" → {feat_path}"
        if args.ids_out:
            msg += f", ids → {args.ids_out}"
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
