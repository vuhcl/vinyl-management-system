#!/usr/bin/env python3
"""
Smoke test: recommender ingest + preprocess on *real* sources.

Examples (from repo root, venv active):

  # Discogs API + local MongoDB (default AOTY path in ingest_all)
  # Token: export DISCOGS_USER_TOKEN=... or put DISCOGS_USER_TOKEN / DISCOGS_TOKEN in repo-root .env
  python scripts/smoke_recommender_ingest.py --username your_discogs_login

  # Custom Mongo
  python scripts/smoke_recommender_ingest.py --username me \\
      --mongo-uri mongodb://localhost:27017 --mongo-db music

  # AOTY from CSV only (no Mongo)
  python scripts/smoke_recommender_ingest.py --username me \\
      --aoty-dir recommender/data/aoty_scraped --no-mongo

  # Mongo only (no Discogs)
  python scripts/smoke_recommender_ingest.py --no-discogs

Discogs ↔ AOTY matching cache: ``{data_dir}/.discogs_cache/`` (throttled HTTP).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _print_df(name: str, df) -> None:
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        print(f"{name}: (not a DataFrame)")
        return
    print(f"{name}: {len(df)} rows × {len(df.columns)} cols", end="")
    if len(df.columns):
        print(f"  cols={list(df.columns)}")
    else:
        print()


def main() -> int:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from shared.project_env import load_project_dotenv

    load_project_dotenv()

    parser = argparse.ArgumentParser(
        description="Run recommender ingest + preprocess on real data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "raw",
        help="Directory for CSV fallbacks and .discogs_cache/",
    )
    parser.add_argument(
        "--username",
        action="append",
        dest="usernames",
        metavar="USER",
        help="Discogs username (repeat for multiple).",
    )
    parser.add_argument(
        "--no-discogs",
        action="store_true",
        help="Do not call Discogs API for collection/wantlist.",
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
        help="MongoDB URI (default: env MONGO_URI or localhost).",
    )
    parser.add_argument(
        "--mongo-db",
        default=os.environ.get("MONGO_DB", "music"),
        help="Mongo database name (default: env MONGO_DB or music).",
    )
    parser.add_argument(
        "--mongo-user-ratings",
        default="user_ratings",
        help="user_ratings collection name.",
    )
    parser.add_argument(
        "--mongo-albums",
        default="albums",
        help="albums collection name.",
    )
    parser.add_argument(
        "--aoty-dir",
        type=Path,
        default=None,
        help="If set, load AOTY from this CSV dir instead of Mongo.",
    )
    parser.add_argument(
        "--no-mongo",
        action="store_true",
        help="With --aoty-dir, disable Mongo; without --aoty-dir, error.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write raw frames to data-dir as collection.csv, etc.",
    )
    parser.add_argument(
        "--release-to-aoty-map",
        type=Path,
        default=None,
        help=(
            "JSON map discogs_release_id → aoty_album_id "
            "(from scripts/build_discogs_aoty_release_map.py)"
        ),
    )
    parser.add_argument(
        "--skip-live-discogs-aoty-mapping",
        action="store_true",
        help="Use only --release-to-aoty-map (no live Discogs master search during ingest)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    from recommender.src.data.ingest import ingest_all
    from recommender.src.data.preprocess import preprocess

    token = (
        os.environ.get("DISCOGS_USER_TOKEN") or os.environ.get("DISCOGS_TOKEN")
    )

    if args.no_discogs:
        discogs: dict | None = None
    else:
        if not args.usernames:
            print(
                "Error: provide --username USER (or use --no-discogs).",
                file=sys.stderr,
            )
            return 2
        if not token:
            print(
                "Error: set DISCOGS_USER_TOKEN (or DISCOGS_TOKEN) in the "
                "environment or repo-root .env (loaded automatically).",
                file=sys.stderr,
            )
            return 2
        discogs = {
            "use_api": True,
            "usernames": list(args.usernames),
            "token": token,
            "release_to_aoty_map_path": args.release_to_aoty_map,
            "skip_live_discogs_aoty_mapping": args.skip_live_discogs_aoty_mapping,
        }

    if args.aoty_dir is not None:
        aoty_dir = Path(args.aoty_dir)
        if not aoty_dir.is_absolute():
            aoty_dir = root / aoty_dir
        if not args.no_mongo and not aoty_dir.exists():
            msg = (
                f"Warning: --aoty-dir {aoty_dir} missing; "
                "CSV ingest may be empty."
            )
            print(msg, file=sys.stderr)
        aoty_scraped: dict = {
            "dir": aoty_dir,
            "use_mongo": False,
            "ratings_file": "ratings.csv",
            "albums_file": "albums.csv",
        }
    else:
        if args.no_mongo:
            print(
                "Error: --no-mongo without --aoty-dir skips AOTY entirely.",
                file=sys.stderr,
            )
            return 2
        aoty_scraped = {
            "use_mongo": True,
            "mongo": {
                "mongo_uri": args.mongo_uri,
                "db_name": args.mongo_db,
                "user_ratings_collection": args.mongo_user_ratings,
                "albums_collection": args.mongo_albums,
            },
        }

    print("=== Smoke: recommender ingest + preprocess ===")
    print(f"data_dir: {data_dir.resolve()}")
    if discogs:
        print(f"Discogs users: {discogs['usernames']}")
    else:
        print("Discogs: off")
    if args.aoty_dir:
        print(f"AOTY: CSV dir {aoty_dir}")
    else:
        print(
            f"AOTY: Mongo {args.mongo_uri!r} db={args.mongo_db!r} "
            f"collections={args.mongo_user_ratings!r}/{args.mongo_albums!r}"
        )

    try:
        raw = ingest_all(data_dir, discogs=discogs, aoty_scraped=aoty_scraped)
    except Exception as e:
        print(f"Ingest failed: {e}", file=sys.stderr)
        return 1

    for key in ("collection", "wantlist", "ratings", "albums"):
        _print_df(f"raw[{key}]", raw[key])

    meta = raw.get("ingest_metadata")
    if meta:
        print("\n=== ingest_metadata (incl. Discogs ↔ AOTY mapping drops) ===")
        print(json.dumps(meta, indent=2, sort_keys=True, default=str))

    if args.write_csv:
        for key in ("collection", "wantlist", "ratings", "albums"):
            df = raw[key]
            if isinstance(df, pd.DataFrame) and not df.empty:
                p = data_dir / f"{key}.csv"
                df.to_csv(p, index=False)
                print(f"Wrote {p}")

    weights = {
        "collection": 1.0,
        "wantlist": 2.0,
        "rating_high": 2.5,
        "rating_mid": 1.5,
        "rating_low_2": 0.8,
        "rating_low_1": 0.4,
    }
    interactions, albums_pp = preprocess(raw, weights)
    _print_df("interactions (merged)", interactions)
    _print_df("albums (for content)", albums_pp)

    if interactions.empty:
        print(
            "\nNo interactions after preprocess. Typical causes:\n"
            "  - Discogs album_id not mapped to any AOTY album_id\n"
            "  - Empty Mongo collections or wrong db/collection names\n"
            "  - Missing CSVs when using --aoty-dir\n",
            file=sys.stderr,
        )
        return 3

    vc = interactions["source"].value_counts()
    print("interactions by source:\n", vc.to_string())
    print(
        f"unique users: {interactions['user_id'].nunique()}  "
        f"unique items: {interactions['album_id'].nunique()}"
    )
    print("\nOK — same path as recommender.pipeline ingest + preprocess.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
