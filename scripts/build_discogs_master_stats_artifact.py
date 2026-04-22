#!/usr/bin/env python3
"""
Build ``artifacts/discogs_master_stats.parquet`` for reranker Tier A + Tier B.

Tier A (broad): aggregates from ``releases_features`` in the Discogs dump
feature store, grouped by ``master_id``, restricted to masters listed in
``discogs_master_to_aoty.json``.

Tier B (sparse): sums / mins from ``marketplace_stats.sqlite`` joined on
``release_id``, rolled up to ``master_id``.

Output columns (one row per mapped AOTY ``album_id``):

  album_id, master_id, release_count, vinyl_release_count,
  unique_country_count, unique_label_count, year_first, year_last, era_span,
  community_want, community_have, num_for_sale, lowest_price,
  has_community_stats

Run from repo root::

  PYTHONPATH=. uv run python scripts/build_discogs_master_stats_artifact.py \\
      --master-json artifacts/discogs_master_to_aoty.json \\
      --output artifacts/discogs_master_stats.parquet
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _create_map_table(conn: sqlite3.Connection, master_ids: list[str]) -> None:
    conn.execute("DROP TABLE IF EXISTS _map_master")
    conn.execute("CREATE TEMP TABLE _map_master (master_id TEXT PRIMARY KEY)")
    conn.executemany(
        "INSERT OR IGNORE INTO _map_master (master_id) VALUES (?)",
        [(m,) for m in master_ids],
    )


def _tier_a_sql(vinyl_expr_sql: str) -> str:
    """Static GROUP BY master_id query; *vinyl_expr_sql* is a SUM()-able int expr."""
    return f"""
        SELECT
            rf.master_id AS master_id,
            COUNT(*) AS release_count,
            SUM(CAST(({vinyl_expr_sql}) AS INTEGER)) AS vinyl_release_count,
            COUNT(DISTINCT NULLIF(TRIM(rf.country), '')) AS unique_country_count,
            COUNT(DISTINCT COALESCE(
                NULLIF(TRIM(json_extract(rf.labels_json, '$[0].id')), ''),
                NULLIF(TRIM(json_extract(rf.labels_json, '$[0].name')), '')
            )) AS unique_label_count,
            MIN(CASE WHEN rf.year > 0 THEN rf.year END) AS year_first,
            MAX(CASE WHEN rf.year > 0 THEN rf.year END) AS year_last
        FROM releases_features AS rf
        INNER JOIN _map_master AS mm ON mm.master_id = rf.master_id
        WHERE rf.master_id IS NOT NULL
          AND TRIM(CAST(rf.master_id AS TEXT)) != ''
        GROUP BY rf.master_id
    """


def _tier_b_sql() -> str:
    return """
        SELECT
            rf.master_id AS master_id,
            SUM(COALESCE(mp.community_want, 0)) AS community_want,
            SUM(COALESCE(mp.community_have, 0)) AS community_have,
            SUM(COALESCE(mp.release_num_for_sale, mp.num_for_sale, 0))
                AS num_for_sale,
            MIN(CASE
                WHEN mp.release_lowest_price IS NOT NULL
                     AND CAST(mp.release_lowest_price AS REAL) > 0
                THEN CAST(mp.release_lowest_price AS REAL)
                END) AS lowest_price,
            MAX(CASE WHEN mp.release_id IS NOT NULL THEN 1 ELSE 0 END)
                AS has_marketplace_row
        FROM releases_features AS rf
        INNER JOIN _map_master AS mm ON mm.master_id = rf.master_id
        LEFT JOIN mpdb.marketplace_stats AS mp
            ON mp.release_id = rf.release_id
        WHERE rf.master_id IS NOT NULL
          AND TRIM(CAST(rf.master_id AS TEXT)) != ''
        GROUP BY rf.master_id
    """


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Discogs dump + marketplace stats per master, join to "
            "AOTY album_id via discogs_master_to_aoty.json."
        )
    )
    parser.add_argument(
        "--master-json",
        type=Path,
        default=Path("artifacts/discogs_master_to_aoty.json"),
        help="discogs_master_id → aoty_album_id JSON",
    )
    parser.add_argument(
        "--feature-db",
        type=Path,
        default=Path("price_estimator/data/feature_store.sqlite"),
        help="SQLite with releases_features",
    )
    parser.add_argument(
        "--marketplace-db",
        type=Path,
        default=Path("price_estimator/data/cache/marketplace_stats.sqlite"),
        help="SQLite with marketplace_stats (optional Tier B)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/discogs_master_stats.parquet"),
    )
    args = parser.parse_args()

    try:
        import pandas as pd
        from price_estimator.src.catalog_proxy import sql_vinyl_preference_key
        from recommender.src.data.discogs_aoty_id_matching import (
            load_master_to_aoty_json,
        )
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    master_json = args.master_json
    if not master_json.is_file():
        print(f"Master map not found: {master_json}", file=sys.stderr)
        return 1

    feat_db = args.feature_db
    if not feat_db.is_absolute():
        feat_db = _REPO_ROOT / feat_db
    if not feat_db.is_file():
        print(f"Feature store not found: {feat_db}", file=sys.stderr)
        return 1

    mp_db = args.marketplace_db
    if not mp_db.is_absolute():
        mp_db = _REPO_ROOT / mp_db

    master_map = load_master_to_aoty_json(master_json)
    if not master_map:
        print("Master map is empty; nothing to aggregate.", file=sys.stderr)
        return 1

    master_ids = sorted({str(k).strip() for k in master_map if str(k).strip()})
    vinyl_sql = sql_vinyl_preference_key("rf.formats_json", "rf.format_desc")

    conn = sqlite3.connect(str(feat_db))
    try:
        conn.row_factory = sqlite3.Row
        _create_map_table(conn, master_ids)
        tier_a = pd.read_sql_query(_tier_a_sql(vinyl_sql), conn)
    finally:
        conn.close()

    if tier_a.empty:
        print("Tier A produced no rows (check master_ids vs releases_features).")
        return 1

    tier_b = None
    if mp_db.is_file():
        # isolation_level=None: autocommit each statement so no deferred
        # transaction keeps mpdb attached after the Tier B SELECT.
        # timeout: brief locks from other readers on marketplace DB.
        conn = sqlite3.connect(
            str(feat_db),
            timeout=120.0,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            _create_map_table(conn, master_ids)
            conn.execute(
                "ATTACH DATABASE ? AS mpdb",
                (str(mp_db.resolve()),),
            )
            try:
                # Avoid pandas read_sql_query here: it can leave the cursor
                # finalized late enough that DETACH raises "database mpdb is locked".
                cur = conn.cursor()
                try:
                    cur.execute(_tier_b_sql())
                    colnames = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                finally:
                    cur.close()
                tier_b = pd.DataFrame.from_records(rows, columns=colnames)
            finally:
                conn.execute("DETACH DATABASE mpdb")
        finally:
            conn.close()
    else:
        print(
            f"Marketplace DB missing ({mp_db}); Tier B columns will be zero.",
            file=sys.stderr,
        )

    tier_a["master_id"] = tier_a["master_id"].astype(str)
    if tier_b is not None and not tier_b.empty:
        tier_b["master_id"] = tier_b["master_id"].astype(str)
        merged = tier_a.merge(tier_b, on="master_id", how="left")
    else:
        merged = tier_a.copy()
        merged["community_want"] = 0
        merged["community_have"] = 0
        merged["num_for_sale"] = 0
        merged["lowest_price"] = None
        merged["has_marketplace_row"] = 0

    merged["community_want"] = merged["community_want"].fillna(0).astype(int)
    merged["community_have"] = merged["community_have"].fillna(0).astype(int)
    merged["num_for_sale"] = merged["num_for_sale"].fillna(0).astype(int)
    merged["has_community_stats"] = (
        merged["has_marketplace_row"].fillna(0).astype(int).clip(0, 1)
    )
    merged["lowest_price"] = merged["lowest_price"].astype(float)

    y0 = merged["year_first"]
    y1 = merged["year_last"]
    span = (y1 - y0).fillna(0).astype(int).clip(lower=0, upper=80)
    merged["era_span"] = span

    inv = {str(mid): str(aid) for mid, aid in master_map.items()}
    merged["album_id"] = merged["master_id"].map(lambda m: inv.get(str(m)))
    merged = merged.dropna(subset=["album_id"]).copy()
    merged["album_id"] = merged["album_id"].astype(str)

    out_cols = [
        "album_id",
        "master_id",
        "release_count",
        "vinyl_release_count",
        "unique_country_count",
        "unique_label_count",
        "year_first",
        "year_last",
        "era_span",
        "community_want",
        "community_have",
        "num_for_sale",
        "lowest_price",
        "has_community_stats",
    ]
    merged = merged[out_cols]

    out_path = args.output
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(
        f"Wrote {len(merged)} rows → {out_path} "
        f"(from {len(master_ids)} master keys in JSON)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
