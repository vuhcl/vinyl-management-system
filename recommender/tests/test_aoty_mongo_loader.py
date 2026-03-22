from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import patch

from shared.aoty.mongo_loader import MongoConfig
from shared.aoty import mongo_loader as mongo_loader_mod


class FakeCursor:
    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = docs

    def limit(self, n: int) -> "FakeCursor":
        return FakeCursor(self._docs[:n])

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = docs

    def find(
        self,
        _query: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ):
        # Ignore projection in tests; loader uses .get() so missing keys
        # are handled gracefully.
        _ = projection
        return FakeCursor(self._docs)


class FakeDatabase:
    def __init__(self, collections: dict[str, FakeCollection]):
        self._collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self._collections[name]


class FakeMongoClient:
    def __init__(self, _uri: str, *, databases: dict[str, FakeDatabase]):
        self._databases = databases

    def __getitem__(self, name: str) -> FakeDatabase:
        return self._databases[name]

    def close(self) -> None:
        return None


def test_load_ratings_from_mongo_skips_missing_score() -> None:
    docs = [
        {"username": "u1", "album_id": 1, "score": "70"},
        {"username": "u2", "album_id": 2, "score": "30"},
        {"username": "u3", "album_id": 3},  # missing score -> skipped
    ]
    fake_db = FakeDatabase(
        {"user_ratings": FakeCollection(docs)},
    )

    def fake_client_factory(uri: str) -> FakeMongoClient:
        return FakeMongoClient(
            uri,
            databases={"music": fake_db},
        )

    cfg = replace(MongoConfig(), mongo_uri="mongodb://fake", db_name="music")
    with patch.object(
        mongo_loader_mod,
        "MongoClient",
        side_effect=fake_client_factory,
    ):
        df = mongo_loader_mod.load_ratings_from_mongo(cfg)

    assert list(df.columns) == ["user_id", "album_id", "rating"]
    assert len(df) == 2
    # 70/20=3.5, 30/20=1.5
    ratings = {row["album_id"]: row["rating"] for row in df.to_dict("records")}
    assert ratings["1"] == 3.5
    assert ratings["2"] == 1.5


def test_load_album_metadata_from_mongo_handles_missing_critic_and_user_score() -> None:
    # Missing critic_score/user_score for at least one album should not raise.
    docs = [
        {
            "album_id": 1549049,
            "artist": "Blackwater Holylight",
            "album": "Not Here Not Gone",
            "critic_score": 79,
            "user_score": 72,
            "release_date": "2026-01-30T00:00:00.000Z",
            "genres": ["Doomgaze", "Heavy Psych"],
            "priority_score": 0.02,
        },
        {
            "album_id": 1680483,
            "artist": "Unknown Artist",
            "album": "Unrated Album",
            # critic_score/user_score intentionally missing
            "release_date": "2020-02-02",
            "genres": [],
        },
    ]
    fake_db = FakeDatabase(
        {"albums": FakeCollection(docs)},
    )

    def fake_client_factory(uri: str) -> FakeMongoClient:
        return FakeMongoClient(
            uri,
            databases={"music": fake_db},
        )

    cfg = replace(MongoConfig(), mongo_uri="mongodb://fake", db_name="music")
    with patch.object(
        mongo_loader_mod,
        "MongoClient",
        side_effect=fake_client_factory,
    ):
        df = mongo_loader_mod.load_album_metadata_from_mongo(cfg)

    assert "album_title" in df.columns
    assert "release_date" in df.columns
    assert "priority_score" in df.columns
    assert set(df["album_id"].tolist()) == {"1549049", "1680483"}

    row_1 = df[df["album_id"] == "1549049"].iloc[0].to_dict()
    assert row_1["album_title"] == "Not Here Not Gone"
    assert "2026" in str(row_1["release_date"])
    assert abs(row_1["priority_score"] - 0.02) < 1e-9
    # avg = (79 + 72) / 2 = 75.5; 75.5 > 10 => /20 => 3.775
    assert abs(row_1["avg_rating"] - (75.5 / 20.0)) < 1e-6

    row_2 = df[df["album_id"] == "1680483"].iloc[0].to_dict()
    assert row_2["avg_rating"] == 0.0
    assert row_2["priority_score"] == 0.0
