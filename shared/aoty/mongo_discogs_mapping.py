"""
Incremental Discogs ↔ AOTY mapping documents stored in MongoDB.

Collections (names configurable on ``MongoConfig``):

- ``discogs_release_master``: ``release_id`` → ``master_id`` (Discogs)
- ``discogs_master_aoty``: ``master_id`` → ``aoty_album_id`` (AOTY / Mongo albums)
- ``discogs_release_aoty``: denormalized ``release_id`` → ``aoty_album_id``

Call :func:`ensure_discogs_mapping_indexes` once (phase A/B scripts do this)
for unique indexes on those lookup keys. :func:`ensure_albums_matching_indexes`
adds ``albums.artist`` for title matching scans.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, MongoClient

from shared.aoty.mongo_loader import MongoConfig


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_discogs_mapping_indexes(mongo: MongoConfig) -> None:
    """
    Idempotent indexes for upserts and lookups on mapping collections.

    - Unique ``release_id`` / ``master_id`` on the respective collections.
    """
    client = MongoClient(mongo.mongo_uri)
    try:
        db = client[mongo.db_name]
        db[mongo.discogs_release_master_collection].create_index(
            [("release_id", ASCENDING)],
            unique=True,
            name="discogs_release_master_release_id_u",
        )
        db[mongo.discogs_master_aoty_collection].create_index(
            [("master_id", ASCENDING)],
            unique=True,
            name="discogs_master_aoty_master_id_u",
        )
        db[mongo.discogs_release_aoty_collection].create_index(
            [("release_id", ASCENDING)],
            unique=True,
            name="discogs_release_aoty_release_id_u",
        )
    finally:
        client.close()


def ensure_albums_matching_indexes(mongo: MongoConfig) -> None:
    """
    Non-unique index on ``albums.artist`` to speed candidate scans for
    :func:`recommender.src.data.discogs_aoty_mongo_match.find_aoty_album_id_for_discogs_master`.

    Case-insensitive anchored regex may still fall back to collection scans in
    some MongoDB versions; this index still helps equality-style matches and
    reduces work when the planner can use it.
    """
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.albums_collection]
        coll.create_index(
            [("artist", ASCENDING)],
            name="albums_artist_1",
        )
    finally:
        client.close()


def upsert_release_master(
    mongo: MongoConfig,
    release_id: str,
    master_id: str | None,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    doc: dict[str, Any] = {
        "release_id": str(release_id),
        "master_id": None if master_id is None else str(master_id),
        "updated_at": _utc_now(),
    }
    if extra:
        doc.update(extra)
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.discogs_release_master_collection]
        coll.update_one(
            {"release_id": doc["release_id"]},
            {"$set": doc},
            upsert=True,
        )
    finally:
        client.close()


def load_release_master_map(mongo: MongoConfig) -> dict[str, str | None]:
    """All stored release → master rows."""
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.discogs_release_master_collection]
        out: dict[str, str | None] = {}
        for doc in coll.find({}, {"release_id": 1, "master_id": 1}):
            rid = doc.get("release_id")
            if rid is None:
                continue
            mid = doc.get("master_id")
            out[str(rid)] = None if mid is None else str(mid)
        return out
    finally:
        client.close()


def upsert_master_aoty(
    mongo: MongoConfig,
    master_id: str,
    aoty_album_id: str | None,
    *,
    status: str = "ok",
    detail: dict[str, Any] | None = None,
) -> None:
    doc: dict[str, Any] = {
        "master_id": str(master_id),
        "aoty_album_id": None if aoty_album_id is None else str(aoty_album_id),
        "status": status,
        "updated_at": _utc_now(),
    }
    if detail:
        doc["detail"] = detail
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.discogs_master_aoty_collection]
        coll.update_one(
            {"master_id": doc["master_id"]},
            {"$set": doc},
            upsert=True,
        )
    finally:
        client.close()


def load_master_aoty_map(mongo: MongoConfig) -> dict[str, str]:
    """master_id → aoty_album_id for rows with a non-empty mapping."""
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.discogs_master_aoty_collection]
        out: dict[str, str] = {}
        for doc in coll.find(
            {"aoty_album_id": {"$exists": True, "$nin": [None, ""]}},
            {"master_id": 1, "aoty_album_id": 1},
        ):
            mid = doc.get("master_id")
            aid = doc.get("aoty_album_id")
            if mid is not None and aid is not None:
                out[str(mid)] = str(aid)
        return out
    finally:
        client.close()


def upsert_release_aoty(
    mongo: MongoConfig,
    release_id: str,
    aoty_album_id: str | None,
) -> None:
    doc = {
        "release_id": str(release_id),
        "aoty_album_id": None if aoty_album_id is None else str(aoty_album_id),
        "updated_at": _utc_now(),
    }
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.discogs_release_aoty_collection]
        coll.update_one(
            {"release_id": doc["release_id"]},
            {"$set": doc},
            upsert=True,
        )
    finally:
        client.close()


def load_release_aoty_map(mongo: MongoConfig) -> dict[str, str]:
    client = MongoClient(mongo.mongo_uri)
    try:
        coll = client[mongo.db_name][mongo.discogs_release_aoty_collection]
        out: dict[str, str] = {}
        for doc in coll.find(
            {"aoty_album_id": {"$exists": True, "$nin": [None, ""]}},
            {"release_id": 1, "aoty_album_id": 1},
        ):
            rid = doc.get("release_id")
            aid = doc.get("aoty_album_id")
            if rid is not None and aid is not None:
                out[str(rid)] = str(aid)
        return out
    finally:
        client.close()
