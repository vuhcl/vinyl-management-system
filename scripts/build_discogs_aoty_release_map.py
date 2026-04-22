#!/usr/bin/env python3
"""
**Deprecated path:** monolithic build (full AOTY→Discogs search + release map).

Prefer the two-phase Mongo-backed pipeline:

- ``scripts/build_discogs_master_to_aoty_artifact.py`` (phase A)
- ``scripts/build_discogs_release_to_aoty_artifact.py`` (phase B)

---

Build and save Discogs release_id → AOTY album_id mapping (JSON artifact).

Use when live Discogs HTTP matching during ingest is flaky; then set in YAML:

  discogs:
    release_to_aoty_map_path: artifacts/discogs_release_to_aoty.json
    skip_live_discogs_aoty_mapping: true

Or omit skip_live to use this file only as a fallback after a live error.

Examples (repo root, venv active):

  .venv/bin/python scripts/build_discogs_aoty_release_map.py \\
      --username nowaki027 --output artifacts/discogs_release_to_aoty.json

  # Use collection.csv / wantlist.csv under data/raw (repo root) instead of API:
  .venv/bin/python scripts/build_discogs_aoty_release_map.py \\
      --from-csv --data-dir data/raw --output artifacts/discogs_release_to_aoty.json

  # Merge new releases into an existing JSON:
  .venv/bin/python scripts/build_discogs_aoty_release_map.py --username me --merge
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
        description="Build Discogs release → AOTY album id map and save JSON.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw data dir (CSVs, .discogs_cache)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/discogs_release_to_aoty.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--username",
        action="append",
        dest="usernames",
        metavar="USER",
        help="Discogs username(s) for collection/wantlist API load",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Load collection/wantlist from data-dir CSVs instead of Discogs API",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Load existing output JSON and only resolve releases not yet present",
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
    )
    parser.add_argument("--mongo-db", default=os.environ.get("MONGO_DB", "music"))
    parser.add_argument(
        "--mongo-albums",
        default="albums",
        help="Mongo albums collection name",
    )
    parser.add_argument(
        "--mongo-user-ratings",
        default="user_ratings",
        help="Mongo user_ratings collection (unused for map build, passed for parity)",
    )
    parser.add_argument(
        "--aoty-dir",
        type=Path,
        default=None,
        help="If set, load AOTY albums from this CSV directory instead of Mongo",
    )
    args = parser.parse_args()

    from shared.project_env import load_project_dotenv

    load_project_dotenv()

    import requests

    from recommender.src.data.discogs_aoty_id_matching import (
        DiscogsHttpHelper,
        DiscogsMatchConfig,
        build_discogs_master_to_aoty_album_id_map,
        build_discogs_release_to_aoty_map,
        load_release_to_aoty_json,
        save_release_to_aoty_json,
    )
    from recommender.src.data.ingest import ingest_all

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    token = os.environ.get("DISCOGS_USER_TOKEN") or os.environ.get("DISCOGS_TOKEN")
    if not token:
        print(
            "Error: set DISCOGS_USER_TOKEN or DISCOGS_TOKEN (.env or env).",
            file=sys.stderr,
        )
        return 2

    if not args.from_csv and not args.usernames:
        print(
            "Error: provide --username USER or --from-csv.",
            file=sys.stderr,
        )
        return 2

    discogs = None
    if args.from_csv:
        discogs = None
    else:
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
            print(f"Error: --aoty-dir is not a directory: {aoty_dir}", file=sys.stderr)
            return 2
        aoty_scraped = {
            "dir": aoty_dir,
            "use_mongo": False,
            "albums_file": "albums.csv",
        }
    else:
        aoty_scraped = {
            "use_mongo": True,
            "mongo": {
                "mongo_uri": args.mongo_uri,
                "db_name": args.mongo_db,
                "user_ratings_collection": args.mongo_user_ratings,
                "albums_collection": args.mongo_albums,
            },
        }

    raw = ingest_all(
        data_dir,
        discogs=discogs,
        aoty_scraped=aoty_scraped,
    )
    albums = raw["albums"]
    collection = raw["collection"]
    wantlist = raw["wantlist"]

    if albums.empty or "album_title" not in albums.columns:
        print("Error: no AOTY albums with album_title for master search.", file=sys.stderr)
        return 3

    release_ids: set[str] = set()
    for df in (collection, wantlist):
        if not df.empty and "album_id" in df.columns:
            release_ids.update(df["album_id"].astype(str).unique())

    if not release_ids:
        print("Error: no release ids in collection/wantlist.", file=sys.stderr)
        return 4

    existing: dict[str, str] = {}
    if args.merge and out_path.is_file():
        existing = load_release_to_aoty_json(out_path)

    to_resolve = sorted(release_ids - set(existing.keys()))
    print(
        f"releases in collection∪wantlist: {len(release_ids)}; "
        f"already in artifact: {len(existing)}; to resolve: {len(to_resolve)}"
    )

    match_cfg = DiscogsMatchConfig()
    cache_dir = data_dir / ".discogs_cache"
    http = DiscogsHttpHelper(
        requests.Session(),
        token,
        match_cfg,
        cache_dir,
    )
    try:
        catalog_stats: dict[str, int] = {}
        master_to_aoty = build_discogs_master_to_aoty_album_id_map(
            albums,
            http=http,
            cfg=match_cfg,
            stats_out=catalog_stats,
        )
        print("catalog_build:", catalog_stats)
        if not master_to_aoty:
            print("Error: empty discogs_master_to_aoty map.", file=sys.stderr)
            return 5

        new_map = {}
        if to_resolve:
            new_map = build_discogs_release_to_aoty_map(
                to_resolve,
                discogs_master_to_aoty=master_to_aoty,
                http=http,
            )
            print(f"newly mapped releases: {len(new_map)}")
        merged = {**existing, **new_map}
        save_release_to_aoty_json(out_path, merged)
        print(f"Wrote {out_path} ({len(merged)} release → aoty entries)")
    finally:
        http.save_disk()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
