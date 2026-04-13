"""Catalog proxy ordering for stats collection queue."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_build_queue_module():
    path = ROOT / "price_estimator" / "scripts" / "build_stats_collection_queue.py"
    spec = importlib.util.spec_from_file_location("build_stats_collection_queue", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _schema_sql() -> str:
    return """
    CREATE TABLE releases_features (
      release_id TEXT PRIMARY KEY,
      master_id TEXT,
      want_count INTEGER,
      have_count INTEGER,
      want_have_ratio REAL,
      genre TEXT,
      style TEXT,
      decade INTEGER,
      year INTEGER,
      country TEXT,
      label_tier INTEGER,
      is_original_pressing INTEGER,
      is_colored_vinyl INTEGER,
      is_picture_disc INTEGER,
      is_promo INTEGER,
      format_desc TEXT,
      artists_json TEXT,
      labels_json TEXT,
      genres_json TEXT,
      styles_json TEXT,
      formats_json TEXT
    );
    """


def test_catalog_proxy_orders_by_master_artist_and_year(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    rows = [
        (
            "1",
            "m1",
            0,
            0,
            None,
            "Rock",
            None,
            1990,
            1990,
            None,
            0,
            0,
            0,
            0,
            0,
            None,
            '[{"id":"10","name":"A"}]',
            None,
            None,
            None,
            None,
        ),
        (
            "2",
            "m1",
            0,
            0,
            None,
            "Rock",
            None,
            1990,
            2000,
            None,
            0,
            0,
            0,
            0,
            0,
            None,
            '[{"id":"10","name":"A"}]',
            None,
            None,
            None,
            None,
        ),
        (
            "3",
            "m2",
            0,
            0,
            None,
            "Rock",
            None,
            1990,
            1980,
            None,
            0,
            0,
            0,
            0,
            0,
            None,
            '[{"id":"20","name":"B"}]',
            None,
            None,
            None,
            None,
        ),
    ]
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        rows,
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=10,
        extra_limit=0,
        rank_by="proxy",
        w_master=1.0,
        w_artist=1.0,
    )
    # m1×2 + artist10×2 = 4 each for 1 and 2; year tie-break 2000 > 1990 → 2 before 1
    assert ordered[:2] == ["2", "1"]
    assert ordered[2] == "3"


def test_max_per_primary_artist_caps_mega_catalog(tmp_path: Path) -> None:
    """High-proxy artist cannot fill the whole head when cap is low."""
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    rows = []
    for i in range(8):
        rid = str(300 + i)
        rows.append(
            (
                rid,
                "m_pop",
                0,
                0,
                None,
                "Rock",
                None,
                1990,
                2000,
                None,
                0,
                0,
                0,
                0,
                0,
                None,
                '[{"id":"777","name":"Mega"}]',
                None,
                None,
                None,
                None,
            ),
        )
    rows.append(
        (
            "199",
            "m_other",
            0,
            0,
            None,
            "Rock",
            None,
            1990,
            1990,
            None,
            0,
            0,
            0,
            0,
            0,
            None,
            '[{"id":"888","name":"Other"}]',
            None,
            None,
            None,
            None,
        ),
    )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        rows,
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    capped = mod.collect_ranked_ids(
        db,
        primary_limit=20,
        extra_limit=0,
        rank_by="proxy",
        w_master=1.0,
        w_artist=1.0,
        max_per_primary_artist=2,
    )
    mega = sum(1 for r in capped if r.startswith("30"))
    other = "199" in capped
    assert mega <= 2
    assert other
    assert len(capped) >= 3

    uncapped = mod.collect_ranked_ids(
        db,
        primary_limit=20,
        extra_limit=0,
        rank_by="proxy",
        w_master=1.0,
        w_artist=1.0,
        max_per_primary_artist=0,
    )
    assert len(uncapped) >= 8


def test_file_format_releases_excluded_from_proxy_queue(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    file_row = (
        "f1",
        "mf",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "File, FLAC",
        '[{"id":"1","name":"A"}]',
        None,
        None,
        None,
        None,
    )
    vinyl_row = (
        "v1",
        "mf",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "Vinyl LP",
        '[{"id":"1","name":"A"}]',
        None,
        None,
        None,
        '[{"name":"Vinyl","qty":"1","descriptions":[]}]',
    )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        [file_row, vinyl_row],
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=10,
        extra_limit=0,
        rank_by="proxy",
        max_per_primary_artist=0,
    )
    assert "f1" not in ordered
    assert "v1" in ordered


def test_unofficial_releases_excluded_from_proxy_queue(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    unofficial_desc = (
        "u1",
        "mf",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "Vinyl, LP, Unofficial Release",
        '[{"id":"1","name":"A"}]',
        None,
        None,
        None,
        None,
    )
    unofficial_json = (
        "u2",
        "mf",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "Vinyl LP",
        '[{"id":"1","name":"A"}]',
        None,
        None,
        None,
        '[{"name":"Vinyl","qty":"1","descriptions":["LP","Unofficial Release"]}]',
    )
    ok_row = (
        "ok_u",
        "mf",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        1999,
        None,
        0,
        0,
        0,
        0,
        0,
        "Vinyl LP",
        '[{"id":"1","name":"A"}]',
        None,
        None,
        None,
        '[{"name":"Vinyl","qty":"1","descriptions":["LP"]}]',
    )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        [unofficial_desc, unofficial_json, ok_row],
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=10,
        extra_limit=0,
        rank_by="proxy",
        max_per_primary_artist=0,
    )
    assert "u1" not in ordered
    assert "u2" not in ordered
    assert "ok_u" in ordered


def test_twelve_inch_format_desc_counts_as_vinyl_without_word_vinyl(
    tmp_path: Path,
) -> None:
    """LP/12\" style lines are vinyl for ordering even if 'vinyl' is absent."""
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    cd = (
        "cd12",
        "ms",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "CD, Album",
        '[{"id":"55","name":"Band"}]',
        None,
        None,
        None,
        '[{"name":"CD","qty":"1","descriptions":[]}]',
    )
    twelve = (
        "tw12",
        "ms",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        '12", 45 RPM, Single',
        '[{"id":"55","name":"Band"}]',
        None,
        None,
        None,
        None,
    )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        [cd, twelve],
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=5,
        extra_limit=0,
        rank_by="proxy",
        max_per_primary_artist=0,
    )
    assert ordered[0] == "tw12"
    assert ordered[1] == "cd12"


def test_target_vinyl_fraction_shapes_proxy_head(tmp_path: Path) -> None:
    """~70% quota: seven vinyl then three CD at same proxy score."""
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    rows: list[tuple] = []
    for i in range(7):
        rows.append(
            (
                f"v{i}",
                "m1",
                0,
                0,
                None,
                "Rock",
                None,
                1990,
                2000,
                None,
                0,
                0,
                0,
                0,
                0,
                "Vinyl, LP",
                '[{"id":"55","name":"Band"}]',
                None,
                None,
                None,
                '[{"name":"Vinyl","qty":"1","descriptions":[]}]',
            ),
        )
    for i in range(10):
        rows.append(
            (
                f"c{i}",
                "m1",
                0,
                0,
                None,
                "Rock",
                None,
                1990,
                2000,
                None,
                0,
                0,
                0,
                0,
                0,
                "CD, Album",
                '[{"id":"55","name":"Band"}]',
                None,
                None,
                None,
                '[{"name":"CD","qty":"1","descriptions":[]}]',
            ),
        )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        rows,
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=10,
        extra_limit=0,
        rank_by="proxy",
        max_per_primary_artist=0,
        target_vinyl_fraction=0.7,
    )
    assert ordered[:7] == [f"v{i}" for i in range(7)]
    assert ordered[7:10] == ["c0", "c1", "c2"]


def test_lp_sorts_before_twelve_inch_at_same_proxy_score(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    twelve = (
        "twelve1",
        "ms",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        '12", 45 RPM, Single',
        '[{"id":"55","name":"Band"}]',
        None,
        None,
        None,
        None,
    )
    lp_row = (
        "lp1",
        "ms",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "Vinyl, LP, Album",
        '[{"id":"55","name":"Band"}]',
        None,
        None,
        None,
        '[{"name":"Vinyl","qty":"1","descriptions":["LP"]}]',
    )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        [twelve, lp_row],
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=5,
        extra_limit=0,
        rank_by="proxy",
        max_per_primary_artist=0,
    )
    assert ordered[0] == "lp1"
    assert ordered[1] == "twelve1"


def test_vinyl_sorts_before_cd_at_same_proxy_score(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    cd = (
        "cd1",
        "ms",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "CD Album",
        '[{"id":"55","name":"Band"}]',
        None,
        None,
        None,
        '[{"name":"CD","qty":"1","descriptions":[]}]',
    )
    vinyl = (
        "vin1",
        "ms",
        0,
        0,
        None,
        "Rock",
        None,
        1990,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        "Vinyl LP",
        '[{"id":"55","name":"Band"}]',
        None,
        None,
        None,
        '[{"name":"Vinyl","qty":"1","descriptions":[]}]',
    )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        [cd, vinyl],
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=5,
        extra_limit=0,
        rank_by="proxy",
        max_per_primary_artist=0,
    )
    assert ordered[0] == "vin1"
    assert ordered[1] == "cd1"


def test_proxy_excludes_various_artists_by_id_and_name(tmp_path: Path) -> None:
    """Various (194), 'various' in name, and primary Unknown Artist are dropped."""
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    va_row = (
        "va194",
        "m9",
        0,
        0,
        None,
        "Comp",
        None,
        2000,
        2000,
        None,
        0,
        0,
        0,
        0,
        0,
        None,
        '[{"id":"194","name":"Various"}]',
        None,
        None,
        None,
        None,
    )
    va_name = (
        "va_name",
        "m9",
        0,
        0,
        None,
        "Comp",
        None,
        2000,
        2001,
        None,
        0,
        0,
        0,
        0,
        0,
        None,
        '[{"id":"999","name":"Various Artists"}]',
        None,
        None,
        None,
        None,
    )
    unknown_primary = (
        "unk1",
        "m9",
        0,
        0,
        None,
        "Rock",
        None,
        2000,
        1995,
        None,
        0,
        0,
        0,
        0,
        0,
        None,
        '[{"id":"777","name":"Unknown Artist"}]',
        None,
        None,
        None,
        None,
    )
    ok_row = (
        "ok1",
        "m9",
        0,
        0,
        None,
        "Rock",
        None,
        2000,
        1990,
        None,
        0,
        0,
        0,
        0,
        0,
        None,
        '[{"id":"1","name":"Real Band"}]',
        None,
        None,
        None,
        None,
    )
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        [va_row, va_name, unknown_primary, ok_row],
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    ordered = mod.collect_ranked_ids(
        db,
        primary_limit=10,
        extra_limit=0,
        rank_by="proxy",
        w_master=1.0,
        w_artist=1.0,
    )
    assert ordered == ["ok1"]


def test_stratified_proxy_picks_top_per_decade(tmp_path: Path) -> None:
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    conn.executemany(
        "INSERT INTO releases_features VALUES (" + ",".join("?" * 21) + ")",
        [
            (
                "10",
                "ma",
                0,
                0,
                None,
                "X",
                None,
                1980,
                1980,
                None,
                0,
                0,
                0,
                0,
                0,
                None,
                '[{"id":"1","name":"Z"}]',
                None,
                None,
                None,
                None,
            ),
            (
                "20",
                "ma",
                0,
                0,
                None,
                "X",
                None,
                1980,
                1981,
                None,
                0,
                0,
                0,
                0,
                0,
                None,
                '[{"id":"1","name":"Z"}]',
                None,
                None,
                None,
                None,
            ),
            (
                "30",
                "mb",
                0,
                0,
                None,
                "Y",
                None,
                1990,
                1990,
                None,
                0,
                0,
                0,
                0,
                0,
                None,
                '[{"id":"2","name":"W"}]',
                None,
                None,
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    out = mod.collect_stratified_ids(
        db,
        per_bucket=1,
        stratify_by="decade",
        seed=1,
        order="proxy",
        w_master=1.0,
        w_artist=1.0,
    )
    assert "20" in out
    assert "30" in out
    decade80 = [r for r in out if r in ("10", "20")]
    assert decade80 == ["20"]


@pytest.mark.parametrize(
    "rank_by,stratify",
    [
        ("combined", "random"),
        ("combined", "community"),
        ("proxy", "community"),
    ],
)
def test_warns_when_community_selected_and_counts_zero(
    tmp_path: Path,
    rank_by: str,
    stratify: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "fs.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(_schema_sql())
    conn.execute(
        "INSERT INTO releases_features VALUES ("
        + ",".join("?" * 21)
        + ")",
        (
            "99",
            None,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    mod = _load_build_queue_module()
    mod.warn_if_community_sort_useless(
        db,
        rank_by=rank_by,
        stratify_order=stratify,
    )
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "MAX(have_count+want_count)" in err
    assert "release_id" in err


def _minimal_feature_row(
    release_id: str,
    *,
    master_id: str | None,
    year: int | None,
    artists_json: str | None,
) -> dict:
    return {
        "release_id": release_id,
        "master_id": master_id,
        "want_count": 0,
        "have_count": 0,
        "want_have_ratio": None,
        "genre": None,
        "style": None,
        "decade": None,
        "year": year,
        "country": None,
        "label_tier": 0,
        "is_original_pressing": 0,
        "is_colored_vinyl": 0,
        "is_picture_disc": 0,
        "is_promo": 0,
        "format_desc": None,
        "artists_json": artists_json,
        "labels_json": None,
        "genres_json": None,
        "styles_json": None,
        "formats_json": None,
    }


def test_feature_store_catalog_proxy_iter(tmp_path: Path) -> None:
    from price_estimator.src.storage.feature_store import FeatureStoreDB

    db = tmp_path / "fs.sqlite"
    store = FeatureStoreDB(db)
    store.upsert_row(
        _minimal_feature_row(
            "1",
            master_id="m1",
            year=1990,
            artists_json='[{"id":"a","name":"X"}]',
        )
    )
    store.upsert_row(
        _minimal_feature_row(
            "2",
            master_id="m1",
            year=2000,
            artists_json='[{"id":"a","name":"X"}]',
        )
    )

    out = list(
        store.iter_release_ids(
            sort_by="catalog_proxy",
            proxy_weight_master=1.0,
            proxy_weight_artist=1.0,
        )
    )
    assert out == ["2", "1"]


def test_marketplace_normalize_extra_fields() -> None:
    from price_estimator.src.storage.marketplace_db import normalize_marketplace_stats

    n = normalize_marketplace_stats(
        {
            "lowest_price": {"value": 1.0, "currency": "USD"},
            "median_price": 2.0,
            "highest_price": 5.0,
            "num_for_sale": 3,
            "blocked_from_sale": True,
        }
    )
    assert n["lowest_price"] == 1.0
    assert n["median_price"] == 2.0
    assert n["highest_price"] == 5.0
    assert n["num_for_sale"] == 3
    assert n["blocked_from_sale"] == 1


def test_merge_release_listing_into_norm_fills_from_release() -> None:
    from price_estimator.src.storage.marketplace_db import (
        merge_release_listing_into_norm,
        normalize_marketplace_stats,
    )

    norm = normalize_marketplace_stats({})
    rel = {
        "lowest_price": {"value": 12.5, "currency": "USD"},
        "num_for_sale": 7,
        "community": {"want": 10, "have": 20},
    }
    out = merge_release_listing_into_norm(norm, {}, rel)
    assert out["lowest_price"] == 12.5
    assert out["median_price"] == 12.5
    assert out["num_for_sale"] == 7


def test_merge_release_listing_respects_stats_payload() -> None:
    from price_estimator.src.storage.marketplace_db import (
        merge_release_listing_into_norm,
        normalize_marketplace_stats,
    )

    payload = {"lowest_price": 9.0, "num_for_sale": 2}
    norm = normalize_marketplace_stats(payload)
    rel = {"lowest_price": {"value": 99.0}, "num_for_sale": 50}
    out = merge_release_listing_into_norm(norm, payload, rel)
    assert out["lowest_price"] == 9.0
    assert out["num_for_sale"] == 2


def test_price_suggestions_ladder_parses_all_grades() -> None:
    import json

    from price_estimator.src.storage.marketplace_db import (
        price_suggestion_values_by_grade,
        price_suggestions_ladder_from_json,
    )

    raw = json.dumps(
        {
            "Near Mint (NM or M-)": {"currency": "USD", "value": 10.0},
            "Very Good Plus (VG+)": {"currency": "USD", "value": 7.5},
            "Poor (P)": {"currency": "USD", "value": 1.0},
        },
        separators=(",", ":"),
    )
    ladder = price_suggestions_ladder_from_json(raw)
    assert set(ladder.keys()) == {
        "Near Mint (NM or M-)",
        "Very Good Plus (VG+)",
        "Poor (P)",
    }
    vals = price_suggestion_values_by_grade(raw)
    assert vals["Near Mint (NM or M-)"] == pytest.approx(10.0)
    assert vals["Very Good Plus (VG+)"] == pytest.approx(7.5)
    assert vals["Poor (P)"] == pytest.approx(1.0)


def test_upsert_preserves_price_suggestions_on_empty_api_payload(tmp_path) -> None:
    import json

    from price_estimator.src.storage.marketplace_db import MarketplaceStatsDB

    db_path = tmp_path / "m.sqlite"
    store = MarketplaceStatsDB(db_path)
    rid = "999001"
    full = {
        "Near Mint (NM or M-)": {"currency": "USD", "value": 12.0},
        "Good (G)": {"currency": "USD", "value": 3.0},
    }
    store.upsert(
        rid,
        {},
        release_payload={"lowest_price": {"value": 5.0}, "num_for_sale": 1},
        price_suggestions_payload=full,
    )
    row1 = store.get(rid)
    assert row1 is not None
    assert json.loads(row1["price_suggestions_json"]) == full

    store.upsert(
        rid,
        {},
        release_payload={"lowest_price": {"value": 6.0}, "num_for_sale": 2},
        price_suggestions_payload={},
    )
    row2 = store.get(rid)
    assert row2 is not None
    assert json.loads(row2["price_suggestions_json"]) == full
