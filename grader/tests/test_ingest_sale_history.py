"""
grader/tests/test_ingest_sale_history.py

Sale-history SQLite → grader JSONL via DiscogsIngester offline.
"""

import random
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from grader.src.data.ingest_sale_history import (
    SALE_HISTORY_SOURCE,
    _allocate_rows_across_media_strata,
    enrich_and_filter_sale_history_records,
    ingest_sale_history_records,
    iter_release_sale_rows,
    sale_row_to_inventory_listing,
    trim_sale_history_records_balanced,
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


def test_sale_row_strips_leading_comments_prefix():
    row = {
        "release_id": "1",
        "row_hash": "h",
        "media_condition": "Very Good Plus (VG+)",
        "sleeve_condition": "Near Mint (NM or M-)",
        "seller_comments": "Comments: plays great",
    }
    listing = sale_row_to_inventory_listing(row)
    assert listing["comments"] == "plays great"
    assert "comments:" not in listing["comments"].lower()


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


def _make_minimal_releases_features(path, release_id: str, format_desc: str) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE releases_features (
            release_id TEXT PRIMARY KEY,
            format_desc TEXT,
            formats_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO releases_features (release_id, format_desc, formats_json) "
        "VALUES (?, ?, ?)",
        (release_id, format_desc, None),
    )
    conn.commit()
    conn.close()


def test_enrich_feature_store_fills_release_format():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fs = root / "fs.sqlite"
        _make_minimal_releases_features(fs, "123", "(Vinyl, LP, Album)")
        cfg: dict = {
            "data": {
                "sale_history": {
                    "feature_store_path": "fs.sqlite",
                    "enrich_from_feature_store": True,
                    "apply_vinyl_filter": False,
                }
            }
        }
        records = [
            {
                "item_id": "123:abc",
                "source": SALE_HISTORY_SOURCE,
                "text": "x" * 50,
            }
        ]
        out, stats = enrich_and_filter_sale_history_records(cfg, root, records)
    assert len(out) == 1
    assert "(Vinyl" in out[0].get("release_format", "")
    assert stats.get("enriched_from_feature_store") == 1


def test_vinyl_filter_drops_cd_with_feature_store():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fs = root / "fs.sqlite"
        _make_minimal_releases_features(fs, "123", "(CD, Album)")
        cfg: dict = {
            "data": {
                "sale_history": {
                    "feature_store_path": "fs.sqlite",
                    "enrich_from_feature_store": True,
                    "apply_vinyl_filter": True,
                }
            }
        }
        records = [
            {
                "item_id": "123:abc",
                "source": SALE_HISTORY_SOURCE,
                "text": "x" * 50,
            }
        ]
        out, stats = enrich_and_filter_sale_history_records(cfg, root, records)
    assert out == []
    assert stats.get("vinyl_dropped", 0) == 1


def _make_multi_release_sale_sqlite(path: Path) -> None:
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
    note = (
        "plays perfectly with light scuff only on one track under bright light"
    )
    for rid in ("2", "1", "3"):
        conn.execute(
            """
            INSERT INTO release_sale (
                release_id, row_hash, order_date, media_condition,
                sleeve_condition, price_original_text, price_user_currency_text,
                seller_comments, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                f"h{rid}",
                "2024-01-01",
                "Very Good Plus (VG+)",
                "Near Mint (NM or M-)",
                "$1",
                "$1",
                note,
                "2024-01-02T00:00:00Z",
            ),
        )
    conn.commit()
    conn.close()


def test_iter_release_sale_rows_ordered_limit(tmp_path: Path) -> None:
    db = tmp_path / "multi.sqlite"
    _make_multi_release_sale_sqlite(db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(
            iter_release_sale_rows(
                conn, limit=2, require_fetch_ok=False, order_random=False
            )
        )
    finally:
        conn.close()
    assert len(rows) == 2
    assert [str(r["release_id"]) for r in rows] == ["1", "2"]


def test_iter_release_sale_rows_require_fetch_ok_and_limit(tmp_path: Path) -> None:
    db = tmp_path / "fetch.sqlite"
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
        CREATE TABLE sale_history_fetch_status (
            release_id TEXT PRIMARY KEY,
            status TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO sale_history_fetch_status (release_id, status) VALUES (?, ?)",
        ("1", "ok"),
    )
    note = (
        "plays perfectly with light scuff only on one track under bright light"
    )
    for rid in ("1", "2"):
        conn.execute(
            """
            INSERT INTO release_sale (
                release_id, row_hash, order_date, media_condition,
                sleeve_condition, price_original_text, price_user_currency_text,
                seller_comments, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                f"h{rid}",
                "2024-01-01",
                "Very Good Plus (VG+)",
                "Near Mint (NM or M-)",
                "$1",
                "$1",
                note,
                "2024-01-02T00:00:00Z",
            ),
        )
    conn.commit()
    conn.row_factory = sqlite3.Row
    rows = list(
        iter_release_sale_rows(
            conn, limit=50, require_fetch_ok=True, order_random=False
        )
    )
    conn.close()
    assert len(rows) == 1
    assert str(rows[0]["release_id"]) == "1"


def test_trim_sale_history_joint_cap_is_deterministic() -> None:
    recs: list[dict] = []
    for i in range(5):
        recs.append(
            {
                "item_id": f"{i}:a",
                "sleeve_label": "Near Mint",
                "media_label": "Very Good Plus",
            }
        )
    out1, st1 = trim_sale_history_records_balanced(
        recs,
        max_rows_per_joint_grade=2,
        joint_sample_seed=12345,
        max_total_sale_history_rows=None,
    )
    out2, _st2 = trim_sale_history_records_balanced(
        recs,
        max_rows_per_joint_grade=2,
        joint_sample_seed=12345,
        max_total_sale_history_rows=None,
    )
    assert st1["strata_capped"] == 1
    assert len(out1) == 2
    assert {r["item_id"] for r in out1} == {r["item_id"] for r in out2}


def test_trim_sale_history_global_cap_sorts_by_item_id() -> None:
    recs = [
        {"item_id": "9:z", "sleeve_label": "Mint", "media_label": "Mint"},
        {"item_id": "10:z", "sleeve_label": "Mint", "media_label": "Mint"},
        {"item_id": "8:z", "sleeve_label": "Mint", "media_label": "Mint"},
    ]
    out, st = trim_sale_history_records_balanced(
        recs,
        max_rows_per_joint_grade=None,
        joint_sample_seed=0,
        max_rows_per_sleeve_grade=None,
        sleeve_stratum_sample_seed=None,
        max_total_sale_history_rows=2,
    )
    assert len(out) == 2
    assert [r["item_id"] for r in out] == ["10:z", "8:z"]
    assert st["after_total_trim"] == 2


def test_trim_sale_history_sleeve_stratum_cap() -> None:
    recs: list[dict] = []
    for i in range(3):
        recs.append(
            {
                "item_id": f"n{i}",
                "sleeve_label": "Near Mint",
                "media_label": "Very Good Plus",
            }
        )
    for i in range(2):
        recs.append(
            {
                "item_id": f"m{i}",
                "sleeve_label": "Near Mint",
                "media_label": "Mint",
            }
        )
    out, st = trim_sale_history_records_balanced(
        recs,
        max_rows_per_joint_grade=None,
        joint_sample_seed=0,
        max_rows_per_sleeve_grade=2,
        sleeve_stratum_sample_seed=777,
        max_total_sale_history_rows=None,
    )
    assert len(out) == 2
    assert all(r["sleeve_label"] == "Near Mint" for r in out)
    assert st.get("sleeve_strata_capped") == 1
    assert st.get("after_joint_trim") == 5
    assert st.get("after_sleeve_trim") == 2


def test_allocate_rows_across_media_strata_respects_small_stratum() -> None:
    rng = random.Random(0)
    counts = {"a": 1, "b": 100}
    got = _allocate_rows_across_media_strata(counts, 50, rng)
    assert sum(got.values()) == 50
    assert got["a"] == 1
    assert got["b"] == 49


def test_trim_sale_history_sleeve_cap_balances_across_media_when_enabled() -> None:
    recs: list[dict] = []
    for i in range(3):
        recs.append(
            {
                "item_id": f"n{i}",
                "sleeve_label": "Near Mint",
                "media_label": "Very Good Plus",
            }
        )
    for i in range(2):
        recs.append(
            {
                "item_id": f"m{i}",
                "sleeve_label": "Near Mint",
                "media_label": "Mint",
            }
        )
    out, st = trim_sale_history_records_balanced(
        recs,
        max_rows_per_joint_grade=None,
        joint_sample_seed=0,
        max_rows_per_sleeve_grade=4,
        sleeve_stratum_sample_seed=999,
        max_total_sale_history_rows=None,
        balance_joint_within_sleeve_trim=True,
    )
    assert len(out) == 4
    assert st.get("balance_joint_within_sleeve_trim") is True
    by_m: dict[str, int] = defaultdict(int)
    for r in out:
        by_m[str(r["media_label"])] += 1
    assert by_m["Mint"] == 2
    assert by_m["Very Good Plus"] == 2
