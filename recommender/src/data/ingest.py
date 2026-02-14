"""
Data ingestion for the recommender subproject.

- **Discogs**: Uses the shared Discogs API (discogs_api) for collection and wantlist.
  All subprojects in vinyl_management_system use the same API client.
- **AOTY**: Reads from scraped data (albumoftheyear.org) via the shared aoty loader.
  Scrapers and scraped data live elsewhere; point config to that directory.

When Discogs token or AOTY path is not set, falls back to CSV files in data_dir.
"""
from pathlib import Path
from typing import cast

import pandas as pd


def _load_collection_csv(path: Path | None, data_dir: Path | None) -> pd.DataFrame:
    """Load collection from CSV. Columns: user_id, album_id (or release_id)."""
    if path is None and data_dir is None:
        return pd.DataFrame(columns=["user_id", "album_id"])
    p = path if path is not None else (data_dir / "collection.csv" if data_dir is not None else None)
    if p is None or not p.exists():
        return pd.DataFrame(columns=["user_id", "album_id"])
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.lower().replace("release_id", "album_id") for c in df.columns})
    if "album_id" not in df.columns and "release_id" in df.columns:
        df["album_id"] = df["release_id"]
    out = cast(pd.DataFrame, df[["user_id", "album_id"]].dropna())
    return out.astype({"user_id": str, "album_id": str})


def _load_wantlist_csv(path: Path | None, data_dir: Path | None) -> pd.DataFrame:
    """Load wantlist from CSV."""
    if path is None and data_dir is None:
        return pd.DataFrame(columns=["user_id", "album_id"])
    p = path if path is not None else (data_dir / "wantlist.csv" if data_dir is not None else None)
    if p is None or not p.exists():
        return pd.DataFrame(columns=["user_id", "album_id"])
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.lower().replace("release_id", "album_id") for c in df.columns})
    out = cast(pd.DataFrame, df[["user_id", "album_id"]].dropna())
    return out.astype({"user_id": str, "album_id": str})


def _load_ratings_csv(path: Path | None, data_dir: Path | None) -> pd.DataFrame:
    """Load ratings from CSV (e.g. exported or sample)."""
    if path is None and data_dir is None:
        return pd.DataFrame(columns=["user_id", "album_id", "rating"])
    p = path if path is not None else (data_dir / "ratings.csv" if data_dir is not None else None)
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
    p = path if path is not None else (data_dir / "albums.csv" if data_dir is not None else None)
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
            from discogs_api import get_user_collection
            import os
            token = discogs_token or os.environ.get("DISCOGS_USER_TOKEN")
            if token:
                dfs = []
                for username in discogs_usernames:
                    df = get_user_collection(username, user_token=token)
                    dfs.append(df)
                if dfs:
                    return pd.concat(dfs, ignore_index=True)
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
            from discogs_api import get_user_wantlist
            import os
            token = discogs_token or os.environ.get("DISCOGS_USER_TOKEN")
            if token:
                dfs = []
                for username in discogs_usernames:
                    df = get_user_wantlist(username, user_token=token)
                    dfs.append(df)
                if dfs:
                    return pd.concat(dfs, ignore_index=True)
        except Exception:
            pass
    return _load_wantlist_csv(path, data_dir)


def load_ratings(
    path: Path | None = None,
    data_dir: Path | None = None,
    *,
    from_aoty_scraped: bool = False,
    aoty_scraped_dir: Path | None = None,
    aoty_ratings_file: str = "ratings.csv",
) -> pd.DataFrame:
    """
    Load AOTY ratings. Prefer scraped data when from_aoty_scraped and aoty_scraped_dir set.
    Otherwise load from CSV (path or data_dir/ratings.csv).
    """
    if from_aoty_scraped and aoty_scraped_dir and Path(aoty_scraped_dir).exists():
        try:
            from aoty import load_ratings_from_scraped
            return load_ratings_from_scraped(Path(aoty_scraped_dir), ratings_file=aoty_ratings_file)
        except Exception:
            pass
    return _load_ratings_csv(path, data_dir)


def load_album_metadata(
    path: Path | None = None,
    data_dir: Path | None = None,
    *,
    from_aoty_scraped: bool = False,
    aoty_scraped_dir: Path | None = None,
    aoty_albums_file: str = "albums.csv",
) -> pd.DataFrame:
    """
    Load album metadata. Prefer AOTY scraped data when from_aoty_scraped and aoty_scraped_dir set.
    Otherwise load from CSV.
    """
    if from_aoty_scraped and aoty_scraped_dir and Path(aoty_scraped_dir).exists():
        try:
            from aoty import load_album_metadata_from_scraped
            return load_album_metadata_from_scraped(Path(aoty_scraped_dir), albums_file=aoty_albums_file)
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
) -> dict[str, pd.DataFrame]:
    """
    Load all raw inputs. Uses shared Discogs API and AOTY scraped data when configured.

    - discogs: optional dict with use_api (bool), usernames (list[str]), token (str, or use env DISCOGS_USER_TOKEN).
    - aoty_scraped: optional dict with dir (Path), ratings_file (str), albums_file (str).
    """
    data_dir = Path(data_dir)
    discogs = discogs or {}
    aoty = aoty_scraped or {}
    use_discogs = discogs.get("use_api", False) and (discogs.get("usernames") or discogs.get("token") or __import__("os").environ.get("DISCOGS_USER_TOKEN"))
    use_aoty = bool(aoty.get("dir") and Path(aoty["dir"]).exists())

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
        from_aoty_scraped=use_aoty,
        aoty_scraped_dir=aoty.get("dir"),
        aoty_ratings_file=aoty.get("ratings_file", "ratings.csv"),
    )
    albums = load_album_metadata(
        path=albums_path,
        data_dir=data_dir,
        from_aoty_scraped=use_aoty,
        aoty_scraped_dir=aoty.get("dir"),
        aoty_albums_file=aoty.get("albums_file", "albums.csv"),
    )

    return {
        "collection": collection,
        "wantlist": wantlist,
        "ratings": ratings,
        "albums": albums,
    }
