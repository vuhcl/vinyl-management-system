"""
grader/tests/test_ingest_sale_history.py

Sale-history SQLite → grader JSONL via DiscogsIngester offline.
"""

import sqlite3
from pathlib import Path

import pytest

from grader.src.data.ingest_sale_history import (
    SALE_HISTORY_SOURCE,
    ingest_sale_history_records,
    sale_row_to_inventory_listing,
)


def _make_sale_sqlite(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE release_sale (
            release_id TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            order_date TEXT NOT NULL,
            media_condition TEXT,
            sleeve_condition TEXT,
            price_original_text TEXT,
            price_user_currency_text TEXT,
            seller_comments TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (release_id, row_hash)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO release_sale (
            release_id, row_hash, order_date, media_condition,
            sleeve_condition, price_original_text, price_user_currency_text,
            seller_comments, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "123",
            "abc",
            "2024-01-01",
            "Very Good Plus (VG+)",
            "Near Mint (NM or M-)",
            "$20.00",
            "$20.00",
            "plays perfectly, light scuff only on one track",
            "2024-01-02T00:00:00Z",
        ),
    )
    conn.commit()
    conn.close()


def test_sale_row_to_inventory_listing_maps_discogs_keys():
    row = {
        "release_id": "9",
        "row_hash": "h1",
        "media_condition": "Mint (M)",
        "sleeve_condition": "Generic Sleeve",
        "seller_comments": "still sealed",
    }
    listing = sale_row_to_inventory_listing(row)
    assert listing["id"] == "9:h1"
    assert listing["condition"] == "Mint (M)"
    assert listing["sleeve_condition"] == "Generic Sleeve"
    assert listing["comments"] == "still sealed"
    assert listing["release"] == {}


def test_ingest_sale_history_records_parses_one_row(
    test_config, guidelines_path, tmp_path, monkeypatch
):
    monkeypatch.setenv("DISCOGS_TOKEN", "TEST_TOKEN")
    db = tmp_path / "sh.sqlite"
    _make_sale_sqlite(db)

    from grader.src.data.ingest_discogs import DiscogsIngester

    ingester = DiscogsIngester(
        test_config,
        guidelines_path,
        offline_parse_only=True,
    )
    records, stats = ingest_sale_history_records(
        db, ingester, limit=None, require_fetch_ok=False
    )
    assert stats["total_rows"] == 1
    assert stats["saved"] == 1
    assert stats["dropped"] == 0
    assert len(records) == 1
    r = records[0]
    assert r["source"] == SALE_HISTORY_SOURCE
    assert r["item_id"] == "123:abc"
    assert r["sleeve_label"] == "Near Mint"
    assert r["media_label"] == "Very Good Plus"
    assert "plays perfectly" in r["text"]


def test_ingest_drops_row_with_missing_notes(test_config, guidelines_path, tmp_path, monkeypatch):
    monkeypatch.setenv("DISCOGS_TOKEN", "TEST_TOKEN")
    db = tmp_path / "sh.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE release_sale (
            release_id TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            order_date TEXT NOT NULL,
            media_condition TEXT,
            sleeve_condition TEXT,
            price_original_text TEXT,
            price_user_currency_text TEXT,
            seller_comments TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (release_id, row_hash)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO release_sale (
            release_id, row_hash, order_date, media_condition,
            sleeve_condition, price_original_text, price_user_currency_text,
            seller_comments, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "1",
            "x",
            "2024-01-01",
            "Very Good Plus (VG+)",
            "Near Mint (NM or M-)",
            "$1",
            "$1",
            "",
            "2024-01-02T00:00:00Z",
        ),
    )
    conn.commit()
    conn.close()

    from grader.src.data.ingest_discogs import DiscogsIngester

    ingester = DiscogsIngester(
        test_config,
        guidelines_path,
        offline_parse_only=True,
    )
    records, stats = ingest_sale_history_records(db, ingester)
    assert stats["saved"] == 0
    assert stats["dropped"] == 1
    assert records == []
