"""
Load AOTY scraped data from a directory produced by the project's scrapers.
Expected: ratings CSV (user_id, album_id, rating), and albums CSV (album_id,
artist, ...).
If your scraper output differs, add an adapter or set column mapping in config.
"""
from pathlib import Path

import pandas as pd


def load_ratings_from_scraped(
    scraped_data_dir: Path,
    ratings_file: str = "ratings.csv",
    *,
    user_col: str = "user_id",
    album_col: str = "album_id",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Load user ratings from AOTY scraped dir.
    Expects user_id, album_id, rating (1–5 or 0–100; normalized to 1–5 if
    needed).
    """
    path = Path(scraped_data_dir) / ratings_file
    if not path.exists():
        return pd.DataFrame(columns=["user_id", "album_id", "rating"])
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # Normalize column names
    col_map = {
        user_col.lower(): "user_id",
        album_col.lower(): "album_id",
        rating_col.lower(): "rating",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    cols = [c for c in ["user_id", "album_id", "rating"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["user_id", "album_id", "rating"])
    df = df[cols].dropna(subset=["user_id", "album_id"])
    if "rating" in df.columns:
        # If ratings are 0–100, scale to 1–5
        if df["rating"].max() > 10:
            df["rating"] = df["rating"] / 20.0
        df["rating"] = df["rating"].clip(0, 5).astype(float)
    else:
        df["rating"] = 0.0
    return df.astype({"user_id": str, "album_id": str, "rating": float})


def load_album_metadata_from_scraped(
    scraped_data_dir: Path,
    albums_file: str = "albums.csv",
    *,
    album_col: str = "album_id",
    artist_col: str = "artist",
    genre_col: str = "genre",
    year_col: str = "year",
    rating_col: str = "avg_rating",
) -> pd.DataFrame:
    """
    Load album metadata from AOTY scraped dir.
    Used for content features (genre, artist, year, avg_rating).
    """
    path = Path(scraped_data_dir) / albums_file
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "album_id",
                "artist",
                "genre",
                "year",
                "avg_rating",
                "album_title",
            ]
        )
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    # Normalize expected/optional title column name.
    album_title_candidates = ("album", "album_title", "title", "name")
    if "album_title" not in df.columns:
        found_title_col = None
        for cand in album_title_candidates:
            if cand in df.columns:
                found_title_col = cand
                break
        if found_title_col is not None:
            df = df.rename(columns={found_title_col: "album_title"})

    col_map = {
        album_col.lower(): "album_id",
        artist_col.lower(): "artist",
        genre_col.lower(): "genre",
        year_col.lower(): "year",
        rating_col.lower(): "avg_rating",
    }
    for old, new in col_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    if "album_title" not in df.columns:
        df["album_title"] = ""

    want = ["album_id", "artist", "genre", "year", "avg_rating", "album_title"]
    have = [c for c in want if c in df.columns]
    return df[have].copy() if have else pd.DataFrame(columns=want)
