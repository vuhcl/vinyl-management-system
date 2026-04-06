"""
Match a Discogs ``master`` (artist + title from ``/masters/{id}``) to an
``albums`` document in Mongo (AOTY album_id).

Used for the **candidate-masters-only** pipeline: we do not iterate all AOTY
albums or run database search per album row.
"""

from __future__ import annotations

import re
from typing import Any

from pymongo import MongoClient

from recommender.src.data.discogs_aoty_id_matching import _title_similarity
from shared.aoty.mongo_loader import MongoConfig


def _album_title_from_doc(doc: dict[str, Any]) -> str:
    for k in ("album_title", "album", "title", "name"):
        v = doc.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _year_from_doc(doc: dict[str, Any]) -> int | None:
    y = doc.get("year")
    if y is None:
        return None
    try:
        yi = int(y)
        return yi if 1800 <= yi <= 2100 else None
    except (TypeError, ValueError):
        return None


def find_aoty_album_id_for_discogs_master(
    mongo: MongoConfig,
    *,
    artist: str,
    album_title: str,
    discogs_year: int | None,
    min_fuzzy: float = 0.35,
) -> str | None:
    """
    Find ``album_id`` in Mongo ``albums`` by case-insensitive artist match +
    fuzzy album title (and light year preference).
    """
    artist = artist.strip()
    album_title = album_title.strip()
    if not artist or not album_title:
        return None

    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.albums_collection]
        artist_esc = re.escape(artist)
        cursor = coll.find(
            {"artist": {"$regex": f"^{artist_esc}$", "$options": "i"}},
            {
                "album_id": 1,
                "artist": 1,
                "album_title": 1,
                "album": 1,
                "title": 1,
                "name": 1,
                "year": 1,
            },
        )
        best_id: str | None = None
        best_score = 0.0
        for doc in cursor:
            aid = doc.get("album_id")
            if aid is None:
                continue
            t = _album_title_from_doc(doc)
            sim = _title_similarity(album_title, t)
            if discogs_year is not None:
                dy = _year_from_doc(doc)
                if dy is not None and dy == discogs_year:
                    sim += 0.08
            if sim > best_score:
                best_score = sim
                best_id = str(aid)
        if best_score >= min_fuzzy and best_id is not None:
            return best_id
        return None
    finally:
        client.close()
