"""
Album metadata helpers for reranker features.

Builds per-album inverted indexes (genres, artists, years, ratings) from
`albums.parquet` + train interactions. Used by the reranker to compute
content-based features (genre Jaccard, artist match, year distance, etc.)
on top of ALS scores. Stage-1 recall is always full-catalog ALS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _split_genres(genre_val: object) -> set[str]:
    """Normalize genre field to a set of lowercase tokens."""
    if genre_val is None or (isinstance(genre_val, float) and np.isnan(genre_val)):
        return set()
    if isinstance(genre_val, list):
        return {str(g).strip().lower() for g in genre_val if str(g).strip()}
    s = str(genre_val).strip()
    if not s:
        return set()
    return {p.strip().lower() for p in s.split(",") if p.strip()}


@dataclass
class RetrievalMetadata:
    """Precomputed indexes for reranker feature computation."""

    album_genres: dict[str, set[str]]
    genre_to_albums: dict[str, set[str]]
    artist_to_albums: dict[str, set[str]]
    album_id_to_artist: dict[str, str]
    album_avg_rating: dict[str, float]
    album_train_count: dict[str, int]
    album_year: dict[str, int]
    album_release_date: dict[str, str]
    album_priority: dict[str, float]
    album_distinct_users: dict[str, int]
    album_rating_rows: dict[str, int]
    valid_album_ids: set[str]


def _empty_meta() -> RetrievalMetadata:
    return RetrievalMetadata(
        album_genres={},
        genre_to_albums={},
        artist_to_albums={},
        album_id_to_artist={},
        album_avg_rating={},
        album_train_count={},
        album_year={},
        album_release_date={},
        album_priority={},
        album_distinct_users={},
        album_rating_rows={},
        valid_album_ids=set(),
    )


def build_retrieval_metadata(
    albums: pd.DataFrame,
    train_interactions: pd.DataFrame,
    *,
    genre_col: str = "genre",
    artist_col: str = "artist",
    album_id_col: str = "album_id",
    avg_rating_col: str = "avg_rating",
    year_col: str = "year",
    release_date_col: str = "release_date",
    priority_col: str = "priority_score",
) -> RetrievalMetadata:
    """
    Build inverted indexes from album metadata + train interaction aggregates.

    `albums` must include at least album_id; optional genre, artist, avg_rating,
    year, release_date, priority_score.
    """
    if albums.empty or album_id_col not in albums.columns:
        return _empty_meta()

    a = albums.copy()
    a[album_id_col] = a[album_id_col].astype(str)

    album_genres: dict[str, set[str]] = {}
    genre_to_albums: dict[str, set[str]] = {}
    artist_to_albums: dict[str, set[str]] = {}
    album_id_to_artist: dict[str, str] = {}
    album_avg_rating: dict[str, float] = {}
    album_year: dict[str, int] = {}
    album_release_date: dict[str, str] = {}
    album_priority: dict[str, float] = {}

    for _, row in a.iterrows():
        aid = str(row[album_id_col])
        if genre_col in a.columns:
            gs = _split_genres(row.get(genre_col))
        else:
            gs = set()
        album_genres[aid] = gs
        for g in gs:
            genre_to_albums.setdefault(g, set()).add(aid)

        if artist_col in a.columns:
            art = str(row.get(artist_col) or "").strip().lower()
            album_id_to_artist[aid] = art
            if art:
                artist_to_albums.setdefault(art, set()).add(aid)
        else:
            album_id_to_artist[aid] = ""

        if avg_rating_col in a.columns:
            v = row.get(avg_rating_col)
            try:
                album_avg_rating[aid] = float(v) if pd.notna(v) else 0.0
            except (TypeError, ValueError):
                album_avg_rating[aid] = 0.0
        else:
            album_avg_rating[aid] = 0.0

        if year_col in a.columns:
            try:
                yv = int(row.get(year_col)) if pd.notna(row.get(year_col)) else 0
            except (TypeError, ValueError):
                yv = 0
            album_year[aid] = max(0, yv)
        else:
            album_year[aid] = 0

        if release_date_col in a.columns:
            rd = row.get(release_date_col)
            album_release_date[aid] = (
                str(rd).strip() if rd is not None and str(rd).strip() else ""
            )
        else:
            album_release_date[aid] = ""

        if priority_col in a.columns:
            try:
                pv = float(row.get(priority_col)) if pd.notna(row.get(priority_col)) else 0.0
            except (TypeError, ValueError):
                pv = 0.0
            album_priority[aid] = pv
        else:
            album_priority[aid] = 0.0

    valid_album_ids = set(a[album_id_col].astype(str))

    g_album = train_interactions.groupby("album_id", sort=False)
    album_train_count = {str(k): int(v) for k, v in g_album.size().items()}
    album_distinct_users = {
        str(k): int(v) for k, v in g_album["user_id"].nunique().items()
    }

    if "source" in train_interactions.columns:
        rmask = train_interactions["source"].astype(str) == "rating"
        sub = train_interactions.loc[rmask]
        if not sub.empty:
            album_rating_rows = {
                str(k): int(v) for k, v in sub.groupby("album_id").size().items()
            }
        else:
            album_rating_rows = {}
    else:
        album_rating_rows = {}

    return RetrievalMetadata(
        album_genres=album_genres,
        genre_to_albums=genre_to_albums,
        artist_to_albums=artist_to_albums,
        album_id_to_artist=album_id_to_artist,
        album_avg_rating=album_avg_rating,
        album_train_count=album_train_count,
        album_year=album_year,
        album_release_date=album_release_date,
        album_priority=album_priority,
        album_distinct_users=album_distinct_users,
        album_rating_rows=album_rating_rows,
        valid_album_ids=valid_album_ids,
    )
