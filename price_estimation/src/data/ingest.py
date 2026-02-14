"""
Data ingestion for the price estimation subproject.

- **Historical sales**: CSV with item_id/release_id, sale_price, sale_date, condition, etc.
  Discogs API does not expose historical sale prices; use CSV from dumps or marketplace exports.
- **Metadata**: Artist, genre, release year, edition from CSV or optional Discogs API.
"""
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to canonical form."""
    df = df.rename(columns={c: c.lower().strip().replace(" ", "_") for c in df.columns})
    # Aliases
    col_map = {}
    if "release_id" in df.columns and "item_id" not in df.columns:
        col_map["release_id"] = "item_id"
    if "id" in df.columns and "item_id" not in df.columns:
        col_map["id"] = "item_id"
    if "price" in df.columns and "sale_price" not in df.columns:
        col_map["price"] = "sale_price"
    if "date" in df.columns and "sale_date" not in df.columns:
        col_map["date"] = "sale_date"
    return df.rename(columns=col_map)


def load_sales_csv(
    path: Path | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load historical sales from CSV.

    Expected columns (case-insensitive):
    - item_id (or release_id, id)
    - sale_price (or price)
    - sale_date (or date) – parseable date
    Optional: sleeve_condition, media_condition, artist, genre, release_year, edition
    """
    if path is None and data_dir is None:
        return pd.DataFrame(
            columns=["item_id", "sale_price", "sale_date"]
        )
    p = path if path is not None else (data_dir / "sales.csv" if data_dir else None)
    if p is None or not p.exists():
        return pd.DataFrame(
            columns=["item_id", "sale_price", "sale_date"]
        )
    df = pd.read_csv(p)
    df = _normalize_columns(df)
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    if "sale_price" in df.columns:
        df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df = df.dropna(subset=["item_id", "sale_price"])
    df["item_id"] = df["item_id"].astype(str)
    return df


def load_metadata_csv(
    path: Path | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load release metadata from CSV (artist, genre, year, edition).

    Expected columns: item_id (or release_id), artist, genre, release_year (or year), edition.
    """
    if path is None and data_dir is None:
        return pd.DataFrame(
            columns=["item_id", "artist", "genre", "release_year"]
        )
    p = path if path is not None else (data_dir / "metadata.csv" if data_dir else None)
    if p is None or not p.exists():
        return pd.DataFrame(
            columns=["item_id", "artist", "genre", "release_year"]
        )
    df = pd.read_csv(p)
    df = _normalize_columns(df)
    if "release_id" in df.columns and "item_id" not in df.columns:
        df = df.rename(columns={"release_id": "item_id"})
    if "year" in df.columns and "release_year" not in df.columns:
        df = df.rename(columns={"year": "release_year"})
    df["item_id"] = df["item_id"].astype(str)
    return df


def fetch_release_metadata_from_discogs(
    release_ids: list[str],
    discogs_token: str | None = None,
) -> pd.DataFrame:
    """
    Fetch release metadata (artist, genre, year) from Discogs API for given release IDs.
    Uses the shared discogs_api client.
    """
    if not release_ids or not discogs_token:
        return pd.DataFrame(columns=["item_id", "artist", "genre", "release_year"])
    try:
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from discogs_api.client import DiscogsClient
    except ImportError:
        return pd.DataFrame(columns=["item_id", "artist", "genre", "release_year"])
    client = DiscogsClient(user_token=discogs_token)
    rows: list[dict[str, Any]] = []
    for rid in release_ids[:500]:
        try:
            data = client._get(f"/releases/{rid}")
            if not isinstance(data, dict):
                continue
            artists = data.get("artists", [])
            artist = artists[0].get("name", "") if artists else ""
            genres = data.get("genres", []) or data.get("styles", [])
            genre = genres[0] if genres else ""
            year = data.get("year") or ""
            rows.append({
                "item_id": str(rid),
                "artist": artist,
                "genre": genre,
                "release_year": year,
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["item_id", "artist", "genre", "release_year"])
    return pd.DataFrame(rows)


def load_sales_and_metadata(
    data_dir: Path | None = None,
    sales_path: Path | None = None,
    metadata_path: Path | None = None,
    *,
    from_discogs: bool = False,
    discogs_token: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical sales and release metadata.

    Returns (sales_df, metadata_df). When from_discogs is True and token is set,
    metadata for unique item_ids in sales is fetched from Discogs; otherwise
    metadata is read from CSV (metadata_path or data_dir/metadata.csv).
    """
    sales = load_sales_csv(path=sales_path, data_dir=data_dir)
    if from_discogs and discogs_token and not sales.empty:
        unique_ids = sales["item_id"].unique().tolist()
        metadata = fetch_release_metadata_from_discogs(unique_ids, discogs_token=discogs_token)
    else:
        metadata = load_metadata_csv(path=metadata_path, data_dir=data_dir)
    return sales, metadata
