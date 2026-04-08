"""Tests for Discogs monthly releases XML → feature rows."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from price_estimator.src.features.vinyliq_features import row_dict_for_inference
from price_estimator.src.ingest.discogs_dump import (
    iter_dump_feature_rows,
    probe_dump_community,
    release_element_to_row,
)
from price_estimator.src.storage.feature_store import FeatureStoreDB


SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<releases>
  <release id="100" status="Accepted">
    <master_id>10</master_id>
    <released>1973-03-00</released>
    <country>US</country>
    <artists>
      <artist><id>501</id><name>Test Artist</name></artist>
    </artists>
    <labels>
      <label id="99" name="Test Label" catno="CAT-1"/>
    </labels>
    <formats>
      <format name="Vinyl" qty="1">
        <descriptions>
          <description>LP</description>
          <description>Promotional</description>
        </descriptions>
      </format>
      <format name="CD" qty="1">
        <descriptions>
          <description>Album</description>
        </descriptions>
      </format>
    </formats>
    <genres><genre>Rock</genre><genre>Pop</genre></genres>
    <styles><style>Prog Rock</style></styles>
    <community><want>5</want><have>10</have></community>
  </release>
  <release id="200" status="Deleted">
    <country>UK</country>
    <community><want>1</want><have>1</have></community>
  </release>
</releases>
"""


def test_release_element_to_row_maps_fields():
    root = ET.fromstring(SAMPLE_XML)
    rel = next(c for c in root if c.get("id") == "100")
    row = release_element_to_row(rel, skip_deleted=True)
    assert row is not None
    assert row["release_id"] == "100"
    assert row["master_id"] == "10"
    assert row["year"] == 1973
    assert row["decade"] == 1970
    assert row["want_count"] == 5
    assert row["have_count"] == 10
    assert row["want_have_ratio"] == 0.5
    assert row["genre"] == "Rock"
    assert row["style"] == "Prog Rock"
    assert row["country"] == "US"
    assert row["is_promo"] == 1
    assert "Vinyl" in (row["format_desc"] or "")
    assert "CD" in (row["format_desc"] or "")

    artists = json.loads(row["artists_json"])
    assert artists == [{"id": "501", "name": "Test Artist"}]
    labels = json.loads(row["labels_json"])
    assert labels == [{"id": "99", "name": "Test Label", "catno": "CAT-1"}]
    assert json.loads(row["genres_json"]) == ["Rock", "Pop"]
    assert json.loads(row["styles_json"]) == ["Prog Rock"]
    fmts = json.loads(row["formats_json"])
    assert len(fmts) == 2
    assert fmts[0]["name"] == "Vinyl"
    assert "LP" in fmts[0]["descriptions"]


def test_skip_deleted_default():
    root = ET.fromstring(SAMPLE_XML)
    deleted = next(c for c in root if c.get("id") == "200")
    assert release_element_to_row(deleted, skip_deleted=True) is None
    row = release_element_to_row(deleted, skip_deleted=False)
    assert row is not None
    assert row["release_id"] == "200"


def test_iter_dump_feature_rows_plain_xml(tmp_path: Path):
    p = tmp_path / "releases.xml"
    p.write_bytes(SAMPLE_XML)
    rows = list(iter_dump_feature_rows(p, skip_deleted=True))
    assert len(rows) == 1
    assert rows[0]["release_id"] == "100"


def test_iter_dump_feature_rows_gzip(tmp_path: Path):
    p = tmp_path / "releases.xml.gz"
    p.write_bytes(gzip.compress(SAMPLE_XML))
    rows = list(iter_dump_feature_rows(p, skip_deleted=False))
    assert len(rows) == 2


def test_probe_dump_community(tmp_path: Path):
    p = tmp_path / "releases.xml"
    p.write_bytes(SAMPLE_XML)
    parsed, nz, mx = probe_dump_community(p, limit=10, skip_deleted=True)
    assert parsed == 1
    assert nz == 1
    assert mx == 15

    parsed2, nz2, mx2 = probe_dump_community(
        p, limit=10, skip_deleted=False
    )
    assert parsed2 == 2
    assert nz2 == 2
    assert mx2 == 15


def test_row_dict_for_inference_catalog_indices_and_counts():
    cat = {
        "want_count": 10,
        "have_count": 5,
        "want_have_ratio": 2.0,
        "year": 1973,
        "decade": 1970,
        "genre": "Rock",
        "country": "US",
        "artists_json": json.dumps([{"id": "1", "name": "A"}]),
        "labels_json": json.dumps([{"id": "2", "name": "L", "catno": "x"}]),
        "genres_json": json.dumps(["Rock", "Pop"]),
        "styles_json": json.dumps(["Prog Rock"]),
        "formats_json": json.dumps(
            [{"name": "Vinyl", "qty": "1", "descriptions": ["LP"]}],
        ),
        "format_desc": "Vinyl LP",
        "is_original_pressing": 0,
        "label_tier": 0,
        "is_colored_vinyl": 0,
        "is_picture_disc": 0,
        "is_promo": 0,
    }
    stats = {"median_price": 0.0, "lowest_price": 0.0, "num_for_sale": 0}
    row = row_dict_for_inference(
        "1",
        "Near Mint (NM or M-)",
        "Near Mint (NM or M-)",
        stats,
        cat,
        genre_index=3.0,
        country_index=2.0,
        primary_artist_index=5.0,
        primary_label_index=7.0,
    )
    assert row["genre_index"] == 3.0
    assert row["country_index"] == 2.0
    assert row["primary_artist_index"] == 5.0
    assert row["primary_label_index"] == 7.0
    assert row["genre_count"] == 2.0
    assert row["style_count"] == 1.0
    assert row["artist_count"] == 1.0
    assert row["label_count"] == 1.0
    assert row["format_count"] == 1.0
    assert row["is_lp"] == 1.0


def test_feature_store_iter_release_ids_sort_by_have(tmp_path: Path):
    db = FeatureStoreDB(tmp_path / "fs.sqlite")
    rows = [
        {
            "release_id": "1",
            "master_id": None,
            "want_count": 5,
            "have_count": 10,
            "want_have_ratio": 0.5,
            "genre": None,
            "style": None,
            "decade": 0,
            "year": 0,
            "country": None,
            "label_tier": 0,
            "is_original_pressing": 0,
            "is_colored_vinyl": 0,
            "is_picture_disc": 0,
            "is_promo": 0,
            "format_desc": None,
            "artists_json": None,
            "labels_json": None,
            "genres_json": None,
            "styles_json": None,
            "formats_json": None,
        },
        {
            "release_id": "2",
            "master_id": None,
            "want_count": 500,
            "have_count": 99,
            "want_have_ratio": 0.0,
            "genre": None,
            "style": None,
            "decade": 0,
            "year": 0,
            "country": None,
            "label_tier": 0,
            "is_original_pressing": 0,
            "is_colored_vinyl": 0,
            "is_picture_disc": 0,
            "is_promo": 0,
            "format_desc": None,
            "artists_json": None,
            "labels_json": None,
            "genres_json": None,
            "styles_json": None,
            "formats_json": None,
        },
    ]
    db.upsert_many(rows)
    ordered = list(db.iter_release_ids(sort_by="have_count"))
    assert ordered == ["2", "1"]
    assert list(db.iter_release_ids(sort_by="want_count")) == ["2", "1"]
    assert list(db.iter_release_ids(sort_by="popularity")) == ["2", "1"]
    assert list(db.iter_release_ids(sort_by="release_id")) == ["1", "2"]
    assert list(db.iter_release_ids(sort_by="want_count", min_want=100)) == ["2"]


def test_feature_store_upsert_many(tmp_path: Path):
    db = FeatureStoreDB(tmp_path / "fs.sqlite")
    rows = [
        {
            "release_id": "1",
            "master_id": None,
            "want_count": 1,
            "have_count": 2,
            "want_have_ratio": 0.5,
            "genre": "a",
            "style": "b",
            "decade": 1990,
            "year": 1995,
            "country": "X",
            "label_tier": 0,
            "is_original_pressing": 0,
            "is_colored_vinyl": 0,
            "is_picture_disc": 0,
            "is_promo": 0,
            "format_desc": None,
            "artists_json": None,
            "labels_json": None,
            "genres_json": None,
            "styles_json": None,
            "formats_json": None,
        },
        {
            "release_id": "2",
            "master_id": "9",
            "want_count": 0,
            "have_count": 0,
            "want_have_ratio": 0.0,
            "genre": None,
            "style": None,
            "decade": 0,
            "year": 0,
            "country": None,
            "label_tier": 0,
            "is_original_pressing": 0,
            "is_colored_vinyl": 0,
            "is_picture_disc": 0,
            "is_promo": 0,
            "format_desc": '12"',
            "artists_json": None,
            "labels_json": None,
            "genres_json": None,
            "styles_json": None,
            "formats_json": None,
        },
    ]
    n = db.upsert_many(rows)
    assert n == 2
    assert db.get("1")["genre"] == "a"
    assert db.get("2")["master_id"] == "9"
