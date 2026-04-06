"""
Data ingestion for the recommender subproject.

- **Discogs**: Uses the shared Discogs API (discogs_api) for collection and wantlist.
  All subprojects in vinyl_management_system use the same API client.
- **AOTY**: Reads from scraped data (albumoftheyear.org) via the shared
  aoty loader.
  Scrapers and scraped data live elsewhere; point config to that directory.

<<<<<<< Updated upstream
When Discogs token or AOTY path is not set, falls back to CSV files in
data_dir.
=======
<<<<<<< Updated upstream
When Discogs token or AOTY path is not set, falls back to CSV files in data_dir.
=======
When Discogs token or AOTY path is not set, falls back to CSV files in
data_dir.

Optional **precomputed** Discogs release → AOTY album id JSON (see
``scripts/build_discogs_aoty_release_map.py``): set
``discogs["release_to_aoty_map_path"]`` and either use
``discogs["skip_live_discogs_aoty_mapping"]=true`` to avoid live HTTP
matching, or rely on automatic fallback when live matching raises.
>>>>>>> Stashed changes
>>>>>>> Stashed changes
"""
from pathlib import Path
from typing import cast

import pandas as pd


def _load_collection_csv(path: Path | None, data_dir: Path | None) -> pd.DataFrame:
    """Load collection from CSV. Columns: user_id, album_id (or release_id)."""
    if path is None and data_dir is None:
        return pd.DataFrame(columns=["user_id", "album_id"])
    if path is not None:
        p = path
    elif data_dir is not None:
        p = data_dir / "collection.csv"
    else:
        p = None
    if p is None or not p.exists():
        return pd.DataFrame(columns=["user_id", "album_id"])
    df = pd.read_csv(p)
    rename_map = {
        c: c.lower().replace("release_id", "album_id") for c in df.columns
    }
    df = df.rename(columns=rename_map)
    if "album_id" not in df.columns and "release_id" in df.columns:
        df["album_id"] = df["release_id"]
    out = cast(pd.DataFrame, df[["user_id", "album_id"]].dropna())
    return out.astype({"user_id": str, "album_id": str})


def _load_wantlist_csv(path: Path | None, data_dir: Path | None) -> pd.DataFrame:
    """Load wantlist from CSV."""
    if path is None and data_dir is None:
        return pd.DataFrame(columns=["user_id", "album_id"])
    if path is not None:
        p = path
    elif data_dir is not None:
        p = data_dir / "wantlist.csv"
    else:
        p = None
    if p is None or not p.exists():
        return pd.DataFrame(columns=["user_id", "album_id"])
    df = pd.read_csv(p)
    rename_map = {
        c: c.lower().replace("release_id", "album_id") for c in df.columns
    }
    df = df.rename(columns=rename_map)
    out = cast(pd.DataFrame, df[["user_id", "album_id"]].dropna())
    return out.astype({"user_id": str, "album_id": str})


def _load_ratings_csv(path: Path | None, data_dir: Path | None) -> pd.DataFrame:
    """Load ratings from CSV (e.g. exported or sample)."""
    if path is None and data_dir is None:
        return pd.DataFrame(columns=["user_id", "album_id", "rating"])
    if path is not None:
        p = path
    elif data_dir is not None:
        p = data_dir / "ratings.csv"
    else:
        p = None
    if p is None or not p.exists():
        return pd.DataFrame(columns=["user_id", "album_id", "rating"])
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    cols = [c for c in ["user_id", "album_id", "rating"] if c in df.columns]
    out = cast(pd.DataFrame, df[cols].dropna())
    return out.astype({"user_id": str, "album_id": str, "rating": float})


def _load_albums_csv(path: Path | None, data_dir: Path | None) -> pd.DataFrame:
    """Load album metadata from CSV."""
    if path is None and data_dir is None:
        return pd.DataFrame(columns=["album_id", "artist", "genre", "year", "avg_rating"])
    if path is not None:
        p = path
    elif data_dir is not None:
        p = data_dir / "albums.csv"
    else:
        p = None
    if p is None or not p.exists():
        return pd.DataFrame(columns=["album_id", "artist", "genre", "year", "avg_rating"])
    df = pd.read_csv(p)
    return df.rename(columns={c: c.lower() for c in df.columns})


def load_collection(
    path: Path | None = None,
    data_dir: Path | None = None,
    *,
    from_discogs: bool = False,
    discogs_usernames: list[str] | None = None,
    discogs_token: str | None = None,
) -> pd.DataFrame:
    """
    Load Discogs collection. Prefer Discogs API when from_discogs and token/usernames set.
    Otherwise load from CSV (path or data_dir/collection.csv).
    """
    if from_discogs and discogs_usernames:
        try:
            from shared.discogs_api import get_user_collection
            import os
            token = discogs_token or os.environ.get("DISCOGS_USER_TOKEN")
            if token:
                dfs = []
                for username in discogs_usernames:
                    df = get_user_collection(username, user_token=token)
                    dfs.append(df)
                if dfs:
                    return pd.concat(
                        dfs,
                        ignore_index=True,
                    )
        except Exception:
            pass
    return _load_collection_csv(path, data_dir)


def load_wantlist(
    path: Path | None = None,
    data_dir: Path | None = None,
    *,
    from_discogs: bool = False,
    discogs_usernames: list[str] | None = None,
    discogs_token: str | None = None,
) -> pd.DataFrame:
    """
    Load Discogs wantlist. Prefer Discogs API when from_discogs and token/usernames set.
    Otherwise load from CSV.
    """
    if from_discogs and discogs_usernames:
        try:
            from shared.discogs_api import get_user_wantlist
            import os
            token = discogs_token or os.environ.get("DISCOGS_USER_TOKEN")
            if token:
                dfs = []
                for username in discogs_usernames:
                    df = get_user_wantlist(username, user_token=token)
                    dfs.append(df)
                if dfs:
                    return pd.concat(
                        dfs,
                        ignore_index=True,
                    )
        except Exception:
            pass
    return _load_wantlist_csv(path, data_dir)


def load_ratings(
    path: Path | None = None,
    data_dir: Path | None = None,
    *,
    from_aoty_scraped: bool = False,
    aoty_scraped_dir: Path | None = None,
    from_aoty_mongo: bool = False,
    aoty_mongo_cfg: dict | None = None,
    aoty_ratings_file: str = "ratings.csv",
) -> pd.DataFrame:
    """
    Load AOTY ratings.

    Priority:
      1) AOTY scraped CSV (when from_aoty_scraped and aoty_scraped_dir exists)
      2) AOTY Mongo (when from_aoty_mongo=True)
      3) CSV fallback (path or data_dir/ratings.csv)
    """
    if from_aoty_scraped and aoty_scraped_dir and Path(aoty_scraped_dir).exists():
        try:
            from shared.aoty import load_ratings_from_scraped
            return load_ratings_from_scraped(
                Path(aoty_scraped_dir),
                ratings_file=aoty_ratings_file,
            )
        except Exception:
            pass

    if from_aoty_mongo:
        try:
            from shared.aoty import MongoConfig, load_ratings_from_mongo

            if aoty_mongo_cfg:
                cfg = MongoConfig(**aoty_mongo_cfg)
            else:
                cfg = MongoConfig()
            return load_ratings_from_mongo(cfg)
        except Exception:
            pass

    return _load_ratings_csv(path, data_dir)


def load_album_metadata(
    path: Path | None = None,
    data_dir: Path | None = None,
    *,
    from_aoty_scraped: bool = False,
    aoty_scraped_dir: Path | None = None,
    from_aoty_mongo: bool = False,
    aoty_mongo_cfg: dict | None = None,
    aoty_albums_file: str = "albums.csv",
) -> pd.DataFrame:
    """
    Load AOTY album metadata.

    Priority:
      1) AOTY scraped CSV (when from_aoty_scraped and aoty_scraped_dir exists)
      2) AOTY Mongo (when from_aoty_mongo=True)
      3) CSV fallback
    """
    if from_aoty_scraped and aoty_scraped_dir and Path(aoty_scraped_dir).exists():
        try:
            from shared.aoty import load_album_metadata_from_scraped
            return load_album_metadata_from_scraped(
                Path(aoty_scraped_dir),
                albums_file=aoty_albums_file,
            )
        except Exception:
            pass

    if from_aoty_mongo:
        try:
            from shared.aoty import MongoConfig, load_album_metadata_from_mongo

            if aoty_mongo_cfg:
                cfg = MongoConfig(**aoty_mongo_cfg)
            else:
                cfg = MongoConfig()
            return load_album_metadata_from_mongo(cfg)
        except Exception:
            pass

    return _load_albums_csv(path, data_dir)


def ingest_all(
    data_dir: Path,
    collection_path: Path | None = None,
    wantlist_path: Path | None = None,
    ratings_path: Path | None = None,
    albums_path: Path | None = None,
    *,
    discogs: dict | None = None,
    aoty_scraped: dict | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Load all raw inputs. Uses shared Discogs API and AOTY scraped data when configured.

    - discogs: optional dict with use_api (bool), usernames (list[str]), token (str, or use env DISCOGS_USER_TOKEN),
      release_to_aoty_map_path (str | Path to JSON), skip_live_discogs_aoty_mapping (bool).
    - aoty_scraped: optional dict with dir (Path), ratings_file (str), albums_file (str).
      When `aoty_scraped.dir` is not set (null), we fall back to loading AOTY data
      from local MongoDB (see `aoty.mongo_loader`) and then fall back to CSV.

    Returns a dict with DataFrames ``collection``, ``wantlist``, ``ratings``, ``albums``
    plus ``ingest_metadata`` (dict). When Discogs→AOTY ID mapping runs,
    ``ingest_metadata["discogs_aoty_mapping"]`` includes catalog-build and per-table
    row/release drop counts.
    """
    data_dir = Path(data_dir)
    discogs = discogs or {}
    aoty = aoty_scraped or {}
    import os

    def _resolve_discogs_token(token: str | None) -> str | None:
        if not token:
            return None
        # If YAML contains "${DISCOGS_TOKEN}" placeholders, don't treat it
        # as a real token (we'll fall back to env vars below).
        if "${" in token and "}" in token:
            return None
        return token

    discogs_token_resolved = (
        _resolve_discogs_token(discogs.get("token"))
        or os.environ.get("DISCOGS_USER_TOKEN")
        or os.environ.get("DISCOGS_TOKEN")
    )

    use_discogs = discogs.get("use_api", False) and (
        discogs.get("usernames")
        or discogs.get("token")
        or discogs_token_resolved
    )
    aoty_dir = aoty.get("dir")
    use_aoty_csv = bool(aoty_dir and Path(aoty_dir).exists())
    # Default: if no scraped dir is set, use local MongoDB.
    use_aoty_mongo = (not use_aoty_csv) and aoty.get("use_mongo", True)

    # Optional nested config override (if you add it to base.yaml later).
    aoty_mongo_cfg = aoty.get("mongo") or aoty.get("mongodb") or aoty.get("mongo_cfg")

    collection = load_collection(
        path=collection_path,
        data_dir=data_dir,
        from_discogs=use_discogs,
        discogs_usernames=discogs.get("usernames"),
        discogs_token=discogs.get("token"),
    )
    wantlist = load_wantlist(
        path=wantlist_path,
        data_dir=data_dir,
        from_discogs=use_discogs,
        discogs_usernames=discogs.get("usernames"),
        discogs_token=discogs.get("token"),
    )
    ratings = load_ratings(
        path=ratings_path,
        data_dir=data_dir,
        from_aoty_scraped=use_aoty_csv,
        aoty_scraped_dir=aoty_dir,
        from_aoty_mongo=use_aoty_mongo,
        aoty_mongo_cfg=aoty_mongo_cfg,
        aoty_ratings_file=aoty.get("ratings_file", "ratings.csv"),
    )
    albums = load_album_metadata(
        path=albums_path,
        data_dir=data_dir,
        from_aoty_scraped=use_aoty_csv,
        aoty_scraped_dir=aoty_dir,
        from_aoty_mongo=use_aoty_mongo,
        aoty_mongo_cfg=aoty_mongo_cfg,
        aoty_albums_file=aoty.get("albums_file", "albums.csv"),
    )

<<<<<<< Updated upstream
=======
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
    # If Discogs data came from the API, `collection`/`wantlist` album_id
    # values are Discogs release IDs. Map them to canonical AOTY album_id so
    # the recommender interactions align across sources.
    ingest_metadata: dict = {"discogs_aoty_mapping": None}
<<<<<<< Updated upstream
    if use_discogs and not albums.empty and discogs_token_resolved:
=======

    def _resolve_release_to_aoty_path(raw_path: str | Path | None) -> Path | None:
        if not raw_path:
            return None
        p = Path(raw_path).expanduser()
        candidates = (
            [p.resolve()]
            if p.is_absolute()
            else [
                (Path.cwd() / p).resolve(),
                (data_dir / p).resolve(),
                (data_dir.parent / p).resolve(),
            ]
        )
        for c in candidates:
            if c.is_file():
                return c
        return None

    release_map_path = _resolve_release_to_aoty_path(
        discogs.get("release_to_aoty_map_path")
    )
    release_map: dict[str, str] = {}
    if release_map_path is not None:
        try:
            from recommender.src.data.discogs_aoty_id_matching import (
                load_release_to_aoty_json,
            )

            release_map = load_release_to_aoty_json(release_map_path)
        except Exception:
            release_map = {}

    def _apply_release_artifact() -> dict:
        from recommender.src.data.discogs_aoty_id_matching import (
            apply_release_to_aoty_map,
        )

        nonlocal collection, wantlist
        c_stats: dict[str, int] = {}
        w_stats: dict[str, int] = {}
        collection = apply_release_to_aoty_map(
            collection, release_map, stats_out=c_stats
        )
        wantlist = apply_release_to_aoty_map(
            wantlist, release_map, stats_out=w_stats
        )
        return {
            "attempted": True,
            "mode": "release_to_aoty_artifact",
            "path": str(release_map_path),
            "catalog_build": {},
            "collection_release_map": dict(c_stats),
            "wantlist_release_map": dict(w_stats),
        }

    skip_live = bool(discogs.get("skip_live_discogs_aoty_mapping"))

    if use_discogs and skip_live:
        if release_map:
            ingest_metadata["discogs_aoty_mapping"] = _apply_release_artifact()
        else:
            ingest_metadata["discogs_aoty_mapping"] = {
                "attempted": True,
                "error": (
                    "skip_live_discogs_aoty_mapping is true but "
                    "release_to_aoty_map_path is missing, unreadable, or empty"
                ),
            }
    elif use_discogs and not albums.empty and discogs_token_resolved:
>>>>>>> Stashed changes
        if "album_title" in albums.columns and (
            albums["album_title"].astype(str).str.strip().ne("").any()
        ):
            try:
                import requests

                from recommender.src.data.discogs_aoty_id_matching import (
                    DiscogsHttpHelper,
                    DiscogsMatchConfig,
                    build_discogs_master_to_aoty_album_id_map,
                    map_discogs_release_ids_to_aoty_album_ids,
                )

                match_cfg = DiscogsMatchConfig()
                cache_dir = data_dir / ".discogs_cache"
                http = DiscogsHttpHelper(
                    requests.Session(),
                    discogs_token_resolved,
                    match_cfg,
                    cache_dir,
                )
                mapping_report: dict = {
                    "attempted": True,
<<<<<<< Updated upstream
=======
                    "mode": "live_discogs_http",
>>>>>>> Stashed changes
                    "catalog_build": {},
                    "collection_release_map": None,
                    "wantlist_release_map": None,
                    "release_mapping_skipped_reason": None,
                }
                catalog_stats: dict[str, int] = {}
                try:
                    discogs_master_to_aoty = (
                        build_discogs_master_to_aoty_album_id_map(
                            albums,
                            http=http,
                            cfg=match_cfg,
                            stats_out=catalog_stats,
                        )
                    )
                    mapping_report["catalog_build"] = dict(catalog_stats)
                    if not discogs_master_to_aoty:
                        mapping_report["release_mapping_skipped_reason"] = (
                            "empty_discogs_master_to_aoty_map"
                        )
                    else:
                        c_stats: dict[str, int] = {}
                        w_stats: dict[str, int] = {}
                        collection = map_discogs_release_ids_to_aoty_album_ids(
                            collection,
                            discogs_master_to_aoty=discogs_master_to_aoty,
                            http=http,
                            cfg=match_cfg,
                            stats_out=c_stats,
                        )
                        wantlist = map_discogs_release_ids_to_aoty_album_ids(
                            wantlist,
                            discogs_master_to_aoty=discogs_master_to_aoty,
                            http=http,
                            cfg=match_cfg,
                            stats_out=w_stats,
                        )
                        mapping_report["collection_release_map"] = dict(c_stats)
                        mapping_report["wantlist_release_map"] = dict(w_stats)
                finally:
                    http.save_disk()
                ingest_metadata["discogs_aoty_mapping"] = mapping_report
            except Exception as exc:
<<<<<<< Updated upstream
                # Mapping is best-effort; downstream preprocess will filter
                # based on available album metadata.
                ingest_metadata["discogs_aoty_mapping"] = {
                    "attempted": True,
                    "error": str(exc),
                }

=======
                if release_map:
                    rep = _apply_release_artifact()
                    rep["live_error"] = str(exc)
                    rep["mode"] = "release_to_aoty_artifact_after_live_error"
                    ingest_metadata["discogs_aoty_mapping"] = rep
                else:
                    ingest_metadata["discogs_aoty_mapping"] = {
                        "attempted": True,
                        "error": str(exc),
                    }

>>>>>>> Stashed changes
>>>>>>> Stashed changes
    return {
        "collection": collection,
        "wantlist": wantlist,
        "ratings": ratings,
        "albums": albums,
        "ingest_metadata": ingest_metadata,
    }
