"""
Ingest jobs: fetch from Discogs (and optionally AOTY) and write to data directories.

Used by the web app after user logs in: trigger ingest for that user's collection/wantlist
so the recommender and other ML components can use the data.
"""
from pathlib import Path

from core.config import get_project_root, load_config


def run_discogs_ingest(
    username: str,
    token: str,
    *,
    data_dir: Path | None = None,
    per_user_dir: bool = True,
) -> dict[str, Path]:
    """
    Fetch collection and wantlist from Discogs for one user and write CSVs.

    If per_user_dir is True, writes to data/raw/{username}/ (so multiple users
    can coexist). Otherwise writes to data/raw/ (single-user mode).

    Returns paths to written files: collection_csv, wantlist_csv.
    """
    from discogs_api import DiscogsClient

    root = get_project_root()
    data_dir = data_dir or (root / "data" / "raw")
    if per_user_dir:
        data_dir = data_dir / username
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    client = DiscogsClient(user_token=token)
    collection = client.collection_to_dataframe(username)
    wantlist = client.wantlist_to_dataframe(username)

    collection_path = data_dir / "collection.csv"
    wantlist_path = data_dir / "wantlist.csv"
    collection.to_csv(collection_path, index=False)
    wantlist.to_csv(wantlist_path, index=False)

    return {"collection_csv": collection_path, "wantlist_csv": wantlist_path}


def run_full_ingest(
    username: str | None = None,
    token: str | None = None,
    *,
    config_path: Path | None = None,
    data_dir: Path | None = None,
    write_csv: bool = True,
) -> dict[str, object]:
    """
    Run the full recommender ingest (Discogs + AOTY) for the given user or config.

    If username/token are provided, they override config for Discogs. When write_csv
    is True, writes collection.csv, wantlist.csv (and ratings/albums if from AOTY)
    to data_dir so the recommender pipeline can use them.
    """
    from recommender.src.data.ingest import ingest_all

    root = get_project_root()
    cfg = load_config(config_path)
    raw_dir = data_dir or (root / cfg.get("paths", {}).get("raw_data", "data/raw"))
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    discogs_cfg = cfg.get("discogs") or {}
    use_api = bool(username and token) or (discogs_cfg.get("use_api") and discogs_cfg.get("usernames"))
    if username and token:
        discogs_cfg = {"use_api": True, "usernames": [username], "token": token}

    aoty_cfg = cfg.get("aoty_scraped") or {}
    aoty_dir = aoty_cfg.get("dir")
    if aoty_dir and not Path(aoty_dir).is_absolute():
        aoty_dir = str(root / aoty_dir)

    raw = ingest_all(
        raw_dir,
        discogs={
            "use_api": use_api,
            "usernames": discogs_cfg.get("usernames") or (username and [username]),
            "token": token or discogs_cfg.get("token"),
        } if use_api else None,
        aoty_scraped={
            "dir": Path(aoty_dir) if aoty_dir else None,
            "ratings_file": aoty_cfg.get("ratings_file", "ratings.csv"),
            "albums_file": aoty_cfg.get("albums_file", "albums.csv"),
        } if aoty_cfg else None,
    )

    if write_csv:
        if not raw["collection"].empty:
            raw["collection"].to_csv(raw_dir / "collection.csv", index=False)
        if not raw["wantlist"].empty:
            raw["wantlist"].to_csv(raw_dir / "wantlist.csv", index=False)
        if not raw["ratings"].empty:
            raw["ratings"].to_csv(raw_dir / "ratings.csv", index=False)
        if not raw["albums"].empty:
            raw["albums"].to_csv(raw_dir / "albums.csv", index=False)

    return {"raw": raw, "data_dir": raw_dir}
