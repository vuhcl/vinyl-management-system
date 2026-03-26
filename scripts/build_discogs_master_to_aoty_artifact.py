#!/usr/bin/env python3
"""
Phase (A): Resolve Discogs **release → master** for collection ∪ wantlist, then
for each **candidate master_id** only, resolve **master → AOTY album_id** via
``GET /masters/{id}`` + Mongo ``albums`` match (no full AOTY catalog scan).

Progress is upserted to Mongo after each release and each master. Writes:

  artifacts/discogs_master_to_aoty.json

Re-run safely: already-mapped rows in Mongo are skipped unless --force.

Example:

  .venv/bin/python scripts/build_discogs_master_to_aoty_artifact.py \\
      --username nowaki027 \\
      --output artifacts/discogs_master_to_aoty.json
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
        description="Phase A: candidate masters → AOTY ids (Mongo + JSON artifact).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/discogs_master_to_aoty.json"),
        help="master_id → aoty_album_id JSON",
    )
    parser.add_argument("--username", action="append", dest="usernames")
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Load collection/wantlist from data-dir CSVs",
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
    )
    parser.add_argument("--mongo-db", default=os.environ.get("MONGO_DB", "music"))
    parser.add_argument("--mongo-albums", default="albums")
    parser.add_argument("--mongo-user-ratings", default="user_ratings")
    parser.add_argument(
        "--mongo-release-master-coll",
        default="discogs_release_master",
        help="Mongo collection for release→master rows",
    )
    parser.add_argument(
        "--mongo-master-aoty-coll",
        default="discogs_master_aoty",
        help="Mongo collection for master→aoty rows",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write JSON artifact after this many new master→aoty mappings",
    )
    parser.add_argument(
        "--force-releases",
        action="store_true",
        help="Re-fetch Discogs release→master even if Mongo has a row",
    )
    parser.add_argument(
        "--force-masters",
        action="store_true",
        help="Re-resolve master→aoty even if Mongo has aoty_album_id",
    )
    parser.add_argument(
        "--min-title-fuzzy",
        type=float,
        default=0.35,
        help="Minimum title similarity vs Mongo album title",
    )
    parser.add_argument(
        "--aoty-dir",
        type=Path,
        default=None,
        help="Load AOTY albums from CSV dir instead of Mongo (not recommended here)",
    )
    parser.add_argument(
        "--skip-ensure-indexes",
        action="store_true",
        help="Do not create Mongo indexes (if you lack db admin rights)",
    )
    args = parser.parse_args()

    from shared.project_env import load_project_dotenv

    load_project_dotenv()

    import requests

    from recommender.src.data.discogs_aoty_id_matching import (
        DiscogsHttpHelper,
        DiscogsMatchConfig,
        load_master_to_aoty_json,
        parse_discogs_master_artist_title_year,
        save_master_to_aoty_json,
    )
    from recommender.src.data.discogs_aoty_mongo_match import (
        find_aoty_album_id_for_discogs_master,
    )
    from recommender.src.data.ingest import ingest_all
    from shared.aoty.mongo_discogs_mapping import (
        ensure_albums_matching_indexes,
        ensure_discogs_mapping_indexes,
        load_master_aoty_map,
        load_release_master_map,
        upsert_master_aoty,
        upsert_release_master,
    )
    from shared.aoty.mongo_loader import MongoConfig

    token = os.environ.get("DISCOGS_USER_TOKEN") or os.environ.get("DISCOGS_TOKEN")
    if not token:
        print("Error: DISCOGS_USER_TOKEN or DISCOGS_TOKEN required.", file=sys.stderr)
        return 2

    if not args.from_csv and not args.usernames:
        print("Error: --username USER or --from-csv required.", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    discogs = None
    if not args.from_csv:
        discogs = {
            "use_api": True,
            "usernames": list(args.usernames or []),
            "token": token,
        }

    if args.aoty_dir is not None:
        aoty_dir = Path(args.aoty_dir)
        if not aoty_dir.is_absolute():
            aoty_dir = _REPO_ROOT / aoty_dir
        if not aoty_dir.is_dir():
            print(f"Error: --aoty-dir not found: {aoty_dir}", file=sys.stderr)
            return 3
        aoty_scraped = {"dir": aoty_dir, "use_mongo": False, "albums_file": "albums.csv"}
    else:
        aoty_scraped = {
            "use_mongo": True,
            "mongo": {
                "mongo_uri": args.mongo_uri,
                "db_name": args.mongo_db,
                "user_ratings_collection": args.mongo_user_ratings,
                "albums_collection": args.mongo_albums,
                "discogs_release_master_collection": args.mongo_release_master_coll,
                "discogs_master_aoty_collection": args.mongo_master_aoty_coll,
            },
        }

    mongo = MongoConfig(
        mongo_uri=args.mongo_uri,
        db_name=args.mongo_db,
        user_ratings_collection=args.mongo_user_ratings,
        albums_collection=args.mongo_albums,
        discogs_release_master_collection=args.mongo_release_master_coll,
        discogs_master_aoty_collection=args.mongo_master_aoty_coll,
    )

    if not args.skip_ensure_indexes and args.aoty_dir is None:
        ensure_discogs_mapping_indexes(mongo)
        ensure_albums_matching_indexes(mongo)
        print("Mongo indexes ensured (discogs mapping + albums.artist).")

    raw = ingest_all(data_dir, discogs=discogs, aoty_scraped=aoty_scraped)
    collection = raw["collection"]
    wantlist = raw["wantlist"]

    release_ids: set[str] = set()
    for df in (collection, wantlist):
        if df.empty or "album_id" not in df.columns:
            continue
        release_ids.update(df["album_id"].astype(str).unique())

    if not release_ids:
        print("Error: no release ids in collection/wantlist.", file=sys.stderr)
        return 4

    print(f"unique releases (collection ∪ wantlist): {len(release_ids)}")

    match_cfg = DiscogsMatchConfig()
    cache_dir = data_dir / ".discogs_cache"
    http = DiscogsHttpHelper(requests.Session(), token, match_cfg, cache_dir)

    stored_rm = load_release_master_map(mongo)
    print(f"Mongo release→master rows (all users): {len(stored_rm)}")

    master_map: dict[str, str] = {}
    if out_path.is_file():
        master_map.update(load_master_to_aoty_json(out_path))
    master_map.update(load_master_aoty_map(mongo))

    try:
        for rid in sorted(release_ids):
            key = str(rid)
            if key in stored_rm and not args.force_releases:
                continue
            data = http.get_release_payload(rid)
            mid_raw = data.get("master_id")
            mid = None if mid_raw is None else str(mid_raw)
            upsert_release_master(mongo, key, mid)
            stored_rm[key] = mid
        http.save_disk()

        master_ids: set[str] = set()
        for rid in release_ids:
            m = stored_rm.get(str(rid))
            if m:
                master_ids.add(str(m))

        print(f"unique candidate master_ids from those releases: {len(master_ids)}")

        new_since_checkpoint = 0
        for mid in sorted(master_ids):
            if master_map.get(mid) and not args.force_masters:
                continue
            payload = http.get_master_payload(mid)
            artist, title, year = parse_discogs_master_artist_title_year(payload)
            aid = find_aoty_album_id_for_discogs_master(
                mongo,
                artist=artist,
                album_title=title,
                discogs_year=year,
                min_fuzzy=args.min_title_fuzzy,
            )
            if aid:
                upsert_master_aoty(
                    mongo,
                    mid,
                    aid,
                    status="ok",
                    detail={"artist": artist, "title": title, "year": year},
                )
                master_map[mid] = aid
                new_since_checkpoint += 1
            else:
                upsert_master_aoty(
                    mongo,
                    mid,
                    None,
                    status="no_mongo_match",
                    detail={"artist": artist, "title": title, "year": year},
                )

            if new_since_checkpoint >= args.checkpoint_every:
                save_master_to_aoty_json(
                    out_path,
                    {k: v for k, v in master_map.items() if v},
                )
                print(f"checkpoint: wrote {out_path} ({len(master_map)} keys)")
                new_since_checkpoint = 0

        http.save_disk()
    finally:
        http.save_disk()

    save_master_to_aoty_json(out_path, {k: v for k, v in master_map.items() if v})
    print(f"Wrote {out_path} ({len([v for v in master_map.values() if v])} mappings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
