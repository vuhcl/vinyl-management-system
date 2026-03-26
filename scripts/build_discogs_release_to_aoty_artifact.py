#!/usr/bin/env python3
"""
Phase (B): Compose **release_id → aoty_album_id** from:

- Mongo ``discogs_release_master`` (release → master), and
- ``discogs_master_to_aoty.json`` and/or Mongo ``discogs_master_aoty``.

Does **not** re-run the expensive phase (A) master resolution.

Example:

  .venv/bin/python scripts/build_discogs_release_to_aoty_artifact.py \\
      --master-json artifacts/discogs_master_to_aoty.json \\
      --output artifacts/discogs_release_to_aoty.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B: release_id → aoty_album_id JSON (+ Mongo upserts).",
    )
    parser.add_argument(
        "--master-json",
        type=Path,
        default=None,
        help="discogs_master_id → aoty_album_id (from phase A)",
    )
    parser.add_argument(
        "--no-mongo-master-map",
        action="store_true",
        help="Do not load master→aoty from Mongo (use --master-json only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/discogs_release_to_aoty.json"),
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
    )
    parser.add_argument("--mongo-db", default=os.environ.get("MONGO_DB", "music"))
    parser.add_argument(
        "--mongo-release-master-coll",
        default="discogs_release_master",
    )
    parser.add_argument(
        "--mongo-master-aoty-coll",
        default="discogs_master_aoty",
    )
    parser.add_argument(
        "--mongo-release-aoty-coll",
        default="discogs_release_aoty",
    )
    parser.add_argument(
        "--skip-ensure-indexes",
        action="store_true",
        help="Do not create Mongo indexes (if you lack db admin rights)",
    )
    args = parser.parse_args()

    from shared.project_env import load_project_dotenv

    load_project_dotenv()

    from recommender.src.data.discogs_aoty_id_matching import (
        load_master_to_aoty_json,
        save_release_to_aoty_json,
    )
    from shared.aoty.mongo_discogs_mapping import (
        ensure_discogs_mapping_indexes,
        load_master_aoty_map,
        load_release_master_map,
        upsert_release_aoty,
    )
    from shared.aoty.mongo_loader import MongoConfig

    mongo = MongoConfig(
        mongo_uri=args.mongo_uri,
        db_name=args.mongo_db,
        discogs_release_master_collection=args.mongo_release_master_coll,
        discogs_master_aoty_collection=args.mongo_master_aoty_coll,
        discogs_release_aoty_collection=args.mongo_release_aoty_coll,
    )

    if not args.skip_ensure_indexes:
        ensure_discogs_mapping_indexes(mongo)
        print("Mongo indexes ensured (discogs mapping collections).")

    master_to_aoty: dict[str, str] = {}
    if not args.no_mongo_master_map:
        master_to_aoty.update(load_master_aoty_map(mongo))
    if args.master_json:
        p = Path(args.master_json).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file():
            master_to_aoty.update(load_master_to_aoty_json(p))

    if not master_to_aoty:
        print(
            "Error: no master→aoty mappings (--master-json and/or "
            "--from-mongo-master-map).",
            file=sys.stderr,
        )
        return 2

    rm = load_release_master_map(mongo)
    if not rm:
        print(
            "Error: Mongo discogs_release_master is empty. Run phase A first.",
            file=sys.stderr,
        )
        return 3

    out: dict[str, str] = {}
    for rid, mid in rm.items():
        if not mid:
            continue
        aid = master_to_aoty.get(str(mid))
        if aid:
            out[str(rid)] = aid

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_release_to_aoty_json(out_path, out)

    # Bulk upsert using a single MongoClient to avoid per-doc connection churn.
    from pymongo import MongoClient, UpdateOne
    from datetime import datetime, timezone

    if out:
        client = MongoClient(mongo.mongo_uri)
        try:
            coll = client[mongo.db_name][mongo.discogs_release_aoty_collection]
            updated_at = datetime.now(timezone.utc)
            # Chunk updates to keep memory bounded.
            chunk_size = 500
            items = list(out.items())
            for i in range(0, len(items), chunk_size):
                chunk = items[i : i + chunk_size]
                ops = [
                    UpdateOne(
                        {"release_id": str(rid)},
                        {
                            "$set": {
                                "release_id": str(rid),
                                "aoty_album_id": str(aid),
                                "updated_at": updated_at,
                            }
                        },
                        upsert=True,
                    )
                    for rid, aid in chunk
                ]
                coll.bulk_write(ops, ordered=False)
        finally:
            client.close()

    print(
        f"Wrote {out_path} ({len(out)} release→aoty); "
        f"Mongo discogs_release_aoty upserted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
