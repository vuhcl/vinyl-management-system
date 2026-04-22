from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from recommender.src.retrieval.dump_master_aoty_join import (
    DumpJoinConfig,
    dump_join_stats,
    fetch_albumish_master_ids,
    load_masters_for_join,
    match_albums_dump_join,
    merge_master_to_aoty_maps,
)


def _write_minimal_feature_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS releases_features;
            DROP TABLE IF EXISTS masters_features;
            CREATE TABLE releases_features (
                release_id TEXT PRIMARY KEY,
                master_id TEXT,
                format_desc TEXT,
                formats_json TEXT
            );
            CREATE TABLE masters_features (
                master_id TEXT PRIMARY KEY,
                primary_artist_name TEXT,
                title TEXT,
                year INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO masters_features VALUES (?,?,?,?)",
            ("9001", "Test Artist", "Hello World", 2000),
        )
        conn.execute(
            "INSERT INTO releases_features VALUES (?,?,?,?)",
            (
                "r1",
                "9001",
                "Vinyl, LP, Album",
                "[]",
            ),
        )
        conn.execute(
            "INSERT INTO masters_features VALUES (?,?,?,?)",
            ("9002", "Other Artist", "No Album Master", 1999),
        )
        conn.execute(
            "INSERT INTO releases_features VALUES (?,?,?,?)",
            ("r2", "9002", "7 Vinyl, Single", "[]"),
        )
        conn.commit()
    finally:
        conn.close()


def test_albumish_filter_and_exact_match(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    _write_minimal_feature_db(db)
    conn = sqlite3.connect(str(db))
    try:
        elig = fetch_albumish_master_ids(conn)
    finally:
        conn.close()
    assert "9001" in elig
    assert "9002" not in elig

    masters = load_masters_for_join(db, elig)
    albums = pd.DataFrame(
        [
            {
                "album_id": "aoty_x",
                "artist": "Test Artist",
                "album_title": "Hello World",
                "year": 2000,
            }
        ]
    )
    cfg = DumpJoinConfig(min_fuzzy_title_similarity=0.99, year_window=1)
    m = match_albums_dump_join(albums, masters, cfg)
    assert m == {"9001": "aoty_x"}


def test_merge_preserves_existing_master(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    _write_minimal_feature_db(db)
    conn = sqlite3.connect(str(db))
    try:
        elig = fetch_albumish_master_ids(conn)
    finally:
        conn.close()
    masters = load_masters_for_join(db, elig)
    albums = pd.DataFrame(
        [
            {
                "album_id": "aoty_x",
                "artist": "Test Artist",
                "album_title": "Hello World",
                "year": 2000,
            }
        ]
    )
    discovered = match_albums_dump_join(
        albums, masters, DumpJoinConfig(min_fuzzy_title_similarity=0.99)
    )
    merged = merge_master_to_aoty_maps({"9001": "prior"}, discovered)
    assert merged["9001"] == "prior"


def test_dump_join_stats() -> None:
    albums = pd.DataFrame(
        {"album_id": ["a", "b"], "artist": ["x", "y"], "album_title": ["t1", "t2"]}
    )
    merged = {"m1": "a"}
    s = dump_join_stats(albums, merged)
    assert s["n_albums"] == 2
    assert s["n_albums_mapped"] == 1
