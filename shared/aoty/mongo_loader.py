"""
Load AOTY scraped data from local MongoDB.

This is the MongoDB counterpart to `aoty/loader.py` (CSV scraped-data loader).
The recommender expects:
  - ratings: user_id, album_id, rating  (rating normalized to 1–5)
  - albums: album_id, artist, genre, year, avg_rating (avg_rating can be 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from pymongo import MongoClient


@dataclass(frozen=True)
class MongoConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "music"
    user_ratings_collection: str = "user_ratings"
    albums_collection: str = "albums"


def _maybe_float(x: Any) -> float | None:
    """
    Best-effort conversion to float.

    Returns None for missing/non-numeric values.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "n/a":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def load_ratings_from_mongo(
    mongo: MongoConfig = MongoConfig(),
    *,
    limit: int | None = None,
    score_field: str = "score",
    username_field: str = "username",
    album_id_field: str = "album_id",
    normalize_to_1_5: bool = True,
) -> pd.DataFrame:
    """
    Load AOTY user ratings from MongoDB.

    Expected document fields (based on DDS project scrapers):
      - username: AOTY username
      - album_id: numeric ID (separate field in your Mongo)
      - score: typically 0–100; normalized to 1–5 for compatibility
    """
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.user_ratings_collection]
        cursor = coll.find(
            {},
            projection={
                username_field: 1,
                album_id_field: 1,
                score_field: 1,
            },
        )
        if limit is not None:
            cursor = cursor.limit(int(limit))
        rows: list[dict[str, Any]] = []
        for doc in cursor:
            username = doc.get(username_field)
            album_id = doc.get(album_id_field)
            rating_raw = _maybe_float(doc.get(score_field))
            if username is None or album_id is None or rating_raw is None:
                continue
            rows.append(
                {
                    "user_id": str(username),
                    "album_id": str(album_id),
                    "rating": float(rating_raw),
                }
            )
    finally:
        client.close()

    df = pd.DataFrame(rows, columns=["user_id", "album_id", "rating"])
    if df.empty:
        return pd.DataFrame(columns=["user_id", "album_id", "rating"])

    # AOTY `score` is typically 0–100; normalize to 0–5 to match our weighting.
    if normalize_to_1_5 and df["rating"].max() > 10:
        df["rating"] = (df["rating"] / 20.0).clip(0, 5)

    df["rating"] = df["rating"].clip(0, 5).astype(float)
    return df.astype({"user_id": str, "album_id": str, "rating": float})


def _format_release_date(value: Any) -> str:
    """Stable string for storage (ISO when possible)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value).strip()


def _extract_year(value: Any) -> int | None:
    """Convert release date-like values to a year integer."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.year)
    if isinstance(value, str):
        # Try common formats; fallback to trailing year digits.
        s = value.strip()
        # e.g. "2020-01-01" or "Aug 2020" or "Feb 13 2020"
        for token in s.replace(",", " ").split():
            if token.isdigit() and len(token) == 4:
                return int(token)
    return None


def load_album_metadata_from_mongo(
    mongo: MongoConfig = MongoConfig(),
    *,
    limit: int | None = None,
    album_id_field: str = "album_id",
    artist_field: str = "artist",
    album_title_field_candidates: tuple[str, ...] = (
        "album",
        "album_title",
        "title",
        "name",
    ),
    critic_score_field: str = "critic_score",
    user_score_field: str = "user_score",
    genres_field: str = "genres",
    release_date_field: str = "release_date",
    avg_rating_field_candidates: tuple[str, ...] = (
        "avg_rating",
        "avg_score",
        "avg_reviewer_score",
    ),
    priority_score_field: str = "priority_score",
) -> pd.DataFrame:
    """
    Load AOTY album metadata from MongoDB.

    Output columns:
      - album_id, artist, album_title, genre, year, avg_rating
      - release_date (str, from Mongo release field; may be empty)
      - priority_score (float; 0.0 if missing)
    """
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.albums_collection]
        projection: dict[str, int] = {
            album_id_field: 1,
            artist_field: 1,
            # Album title is used for Discogs matching.
            genres_field: 1,
            release_date_field: 1,
        }
        for cand in album_title_field_candidates:
            projection[cand] = 1
        for cand in avg_rating_field_candidates:
            projection[cand] = 1
        # DS Group Project schema uses critic_score/user_score; use them to
        # compute an avg rating if explicit avg_rating isn't present.
        projection[critic_score_field] = 1
        projection[user_score_field] = 1
        projection[priority_score_field] = 1

        cursor = coll.find({}, projection=projection)
        if limit is not None:
            cursor = cursor.limit(int(limit))

        rows: list[dict[str, Any]] = []
        for doc in cursor:
            album_id = doc.get(album_id_field)
            artist = doc.get(artist_field) or ""
            album_title = ""
            for cand in album_title_field_candidates:
                val = doc.get(cand)
                if val:
                    album_title = str(val)
                    break
            genres_val = doc.get(genres_field) or []
            rd_raw = doc.get(release_date_field)
            release_date_str = _format_release_date(rd_raw)
            year = _extract_year(rd_raw)

            # avg rating is optional; try explicit avg fields first, then
            # fall back to (critic_score + user_score) / 2.
            rating_raw: float | None = None
            for cand in avg_rating_field_candidates:
                maybe = _maybe_float(doc.get(cand))
                if maybe is not None:
                    rating_raw = float(maybe)
                    break
            if rating_raw is None:
                critic = _maybe_float(doc.get(critic_score_field))
                user = _maybe_float(doc.get(user_score_field))
                if critic is not None and user is not None:
                    rating_raw = (critic + user) / 2.0
                elif critic is not None:
                    rating_raw = critic
                elif user is not None:
                    rating_raw = user
            avg_rating = float(rating_raw) if rating_raw is not None else 0.0

            genre_str: str
            if isinstance(genres_val, list):
                genre_str = ",".join(
                    [
                        str(g).strip()
                        for g in genres_val
                        if g is not None and str(g).strip()
                    ]
                )
            else:
                genre_str = str(genres_val) if genres_val is not None else ""

            if album_id is None:
                continue
            if year is None:
                year = 0

            # If avg_rating is on 0–100 scale, normalize.
            if avg_rating > 10:
                avg_rating = float((avg_rating / 20.0))
            avg_rating = float(max(0.0, min(5.0, avg_rating)))

            pri = _maybe_float(doc.get(priority_score_field))
            priority_score = float(pri) if pri is not None else 0.0

            rows.append(
                {
                    "album_id": str(album_id),
                    "artist": str(artist),
                    "album_title": album_title,
                    "genre": genre_str,
                    "year": int(year),
                    "avg_rating": avg_rating,
                    "release_date": release_date_str,
                    "priority_score": priority_score,
                }
            )
    finally:
        client.close()

    want_cols = [
        "album_id",
        "artist",
        "album_title",
        "genre",
        "year",
        "avg_rating",
        "release_date",
        "priority_score",
    ]
    df = pd.DataFrame(rows, columns=want_cols)
    if df.empty:
        return pd.DataFrame(columns=want_cols)
    return df.astype(
        {
            "album_id": str,
            "artist": str,
            "album_title": str,
            "genre": str,
            "year": int,
            "avg_rating": float,
            "release_date": str,
            "priority_score": float,
        }
    )
