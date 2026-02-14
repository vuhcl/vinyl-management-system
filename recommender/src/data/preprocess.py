"""
Data cleaning and normalization. Build unified interactions with weights.
"""
from pathlib import Path

import pandas as pd


def apply_weights(
    collection: pd.DataFrame,
    wantlist: pd.DataFrame,
    ratings: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Merge collection, wantlist, ratings into one interaction table with strength.
    Weights: collection, wantlist, rating_high (>=4), rating_mid (==3).
    """
    rows = []

    if not collection.empty:
        c = collection[["user_id", "album_id"]].copy()
        c["strength"] = weights.get("collection", 1.0)
        c["source"] = "collection"
        rows.append(c)

    if not wantlist.empty:
        w = wantlist[["user_id", "album_id"]].copy()
        w["strength"] = weights.get("wantlist", 2.0)
        w["source"] = "wantlist"
        rows.append(w)

    if not ratings.empty and "rating" in ratings.columns:
        r = ratings.copy()
        r["strength"] = r["rating"].map(
            lambda x: weights.get("rating_high", 2.5) if x >= 4 else (weights.get("rating_mid", 1.5) if x == 3 else 0.0)
        )
        r = r[r["strength"] > 0][["user_id", "album_id", "strength"]]
        r["source"] = "rating"
        rows.append(r)

    if not rows:
        return pd.DataFrame(columns=["user_id", "album_id", "strength", "source"])

    out = pd.concat(rows, ignore_index=True)
    # Aggregate same user-album from multiple sources: take max strength
    out = out.groupby(["user_id", "album_id"], as_index=False).agg({"strength": "max", "source": "first"})
    return out


def normalize_ids(interactions: pd.DataFrame, albums: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure album_id types match and filter to albums we have metadata for (if provided)."""
    interactions = interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["album_id"] = interactions["album_id"].astype(str)

    if not albums.empty and "album_id" in albums.columns:
        valid_ids = set(albums["album_id"].astype(str))
        interactions = interactions[interactions["album_id"].isin(valid_ids)]

    if not albums.empty:
        albums = albums.copy()
        albums["album_id"] = albums["album_id"].astype(str)
    return interactions, albums


def preprocess(
    raw: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and normalize. Returns (interactions, albums).
    """
    collection = raw.get("collection", pd.DataFrame())
    wantlist = raw.get("wantlist", pd.DataFrame())
    ratings = raw.get("ratings", pd.DataFrame())
    albums = raw.get("albums", pd.DataFrame())

    interactions = apply_weights(collection, wantlist, ratings, weights)
    interactions, albums = normalize_ids(interactions, albums)
    return interactions, albums


def save_processed(interactions: pd.DataFrame, albums: pd.DataFrame, out_dir: Path) -> None:
    """Write processed interactions and albums to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    albums.to_parquet(out_dir / "albums.parquet", index=False)


def load_processed(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed interactions and albums from out_dir."""
    out_dir = Path(out_dir)
    interactions = pd.read_parquet(out_dir / "interactions.parquet")
    albums = pd.read_parquet(out_dir / "albums.parquet")
    return interactions, albums
