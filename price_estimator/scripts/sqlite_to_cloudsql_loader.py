#!/usr/bin/env python3
"""Bulk-load SQLite releases_features + marketplace_stats into Postgres.

Run from repo root::

  uv run python price_estimator/scripts/sqlite_to_cloudsql_loader.py \\
    --feature-store price_estimator/data/feature_store.sqlite \\
    --marketplace-db price_estimator/data/cache/marketplace_stats.sqlite \\
    --database-url \"$DATABASE_URL\"

Needs ``cloud-sql-proxy`` (or similar) where ``DATABASE_URL`` points.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import psycopg

from price_estimator.src.storage.feature_store import RELEASES_FEATURES_COLUMNS

_MARKETPLACE_COLUMNS: tuple[str, ...] = (
    "release_id",
    "fetched_at",
    "num_for_sale",
    "blocked_from_sale",
    "raw_json",
    "release_raw_json",
    "price_suggestions_json",
    "release_lowest_price",
    "release_num_for_sale",
    "community_want",
    "community_have",
)

_INSERT_MARKETPLACE_SQL = """
INSERT INTO marketplace_stats (
    release_id, fetched_at, num_for_sale, blocked_from_sale,
    raw_json, release_raw_json, price_suggestions_json,
    release_lowest_price, release_num_for_sale,
    community_want, community_have
) VALUES (
    %(release_id)s, %(fetched_at)s, %(num_for_sale)s,
    %(blocked_from_sale)s, %(raw_json)s, %(release_raw_json)s,
    %(price_suggestions_json)s, %(release_lowest_price)s,
    %(release_num_for_sale)s, %(community_want)s,
    %(community_have)s
)
ON CONFLICT (release_id) DO UPDATE SET
    fetched_at = EXCLUDED.fetched_at,
    num_for_sale = EXCLUDED.num_for_sale,
    blocked_from_sale = EXCLUDED.blocked_from_sale,
    raw_json = EXCLUDED.raw_json,
    release_raw_json = EXCLUDED.release_raw_json,
    price_suggestions_json = EXCLUDED.price_suggestions_json,
    release_lowest_price = EXCLUDED.release_lowest_price,
    release_num_for_sale = EXCLUDED.release_num_for_sale,
    community_want = EXCLUDED.community_want,
    community_have = EXCLUDED.community_have
"""


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {str(r[1]) for r in cur.fetchall()}


def load_releases_features(
    sqlite_path: Path,
    pg_conn: psycopg.Connection,
    *,
    log_every: int,
    copy_chunk_rows: int,
) -> int:
    cols = RELEASES_FEATURES_COLUMNS
    src = sqlite3.connect(str(sqlite_path))
    src.row_factory = sqlite3.Row
    try:
        existing = _sqlite_table_columns(src, "releases_features")
        if "release_id" not in existing:
            logging.warning(
                "SQLite missing releases_features.release_id; skip features load"
            )
            return 0

        cols_csv = ", ".join(cols)
        sel_cols = ", ".join(c for c in cols if c in existing)
        q = f"SELECT {sel_cols} FROM releases_features"

        stmt = f"COPY releases_features ({cols_csv}) FROM STDIN"
        n = 0
        chunk: list[tuple] = []
        chunk_cap = max(1, int(copy_chunk_rows))

        def flush_chunk() -> None:
            if not chunk:
                return
            with pg_conn.cursor() as cur:
                with cur.copy(stmt) as copy:
                    for t in chunk:
                        copy.write_row(t)
            pg_conn.commit()
            chunk.clear()

        try:
            for row in src.execute(q):
                rd = {str(k): row[k] for k in row.keys()}
                chunk.append(tuple(rd.get(c) for c in cols))
                n += 1
                if len(chunk) >= chunk_cap:
                    flush_chunk()
                if log_every > 0 and n % log_every == 0:
                    logging.info(
                        "releases_features COPY progress: %s rows",
                        n,
                    )
            flush_chunk()
        finally:
            chunk.clear()

        logging.info("releases_features loaded: %s rows", n)
        return n
    finally:
        src.close()


def _marketplace_row(row: sqlite3.Row, existing: set[str]) -> dict:
    rd = {c: None for c in _MARKETPLACE_COLUMNS}
    keys = set(row.keys())
    for c in _MARKETPLACE_COLUMNS:
        if c in existing and c in keys:
            rd[c] = row[c]
    if rd["raw_json"] is None:
        rd["raw_json"] = "{}"
    if rd["fetched_at"] is None:
        rd["fetched_at"] = ""
    return rd


def load_marketplace_stats(
    sqlite_path: Path,
    pg_conn: psycopg.Connection,
    *,
    batch_size: int,
    log_every: int,
) -> int:
    src = sqlite3.connect(str(sqlite_path))
    src.row_factory = sqlite3.Row
    try:
        existing = _sqlite_table_columns(src, "marketplace_stats")
        if "release_id" not in existing:
            logging.warning(
                "SQLite missing marketplace_stats.release_id; skip marketplace load"
            )
            return 0

        n = 0
        batch: list[dict] = []
        with pg_conn.cursor() as cur:
            for row in src.execute("SELECT * FROM marketplace_stats"):
                batch.append(_marketplace_row(row, existing))
                if len(batch) >= batch_size:
                    cur.executemany(_INSERT_MARKETPLACE_SQL, batch)
                    n += len(batch)
                    batch.clear()
                    pg_conn.commit()
                    if log_every > 0 and n % log_every == 0:
                        logging.info(
                            "marketplace_stats upsert progress: %s rows",
                            n,
                        )
            if batch:
                cur.executemany(_INSERT_MARKETPLACE_SQL, batch)
                n += len(batch)
                pg_conn.commit()
        logging.info("marketplace_stats loaded: %s rows", n)
        return n
    finally:
        src.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-store", type=Path, required=True)
    p.add_argument("--marketplace-db", type=Path, required=True)
    p.add_argument(
        "--database-url",
        required=True,
        help="Postgres DSN (via cloud-sql-proxy)",
    )
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument(
        "--log-every",
        type=int,
        default=100_000,
        help="Progress log interval (0 disables)",
    )
    p.add_argument(
        "--copy-chunk-rows",
        type=int,
        default=50_000,
        help=(
            "Commit after this many releases_features rows per COPY "
            "(smaller = safer over flaky proxy / long loads)"
        ),
    )
    p.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip releases_features (e.g. after partial load + TRUNCATE retry)",
    )
    p.add_argument(
        "--skip-marketplace",
        action="store_true",
        help="Skip marketplace_stats load",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fs_path = args.feature_store.expanduser().resolve()
    mp_path = args.marketplace_db.expanduser().resolve()
    if not fs_path.is_file():
        logging.error("feature-store file not found: %s", fs_path)
        return 1
    if not mp_path.is_file():
        logging.error("marketplace-db file not found: %s", mp_path)
        return 1

    with psycopg.connect(args.database_url) as conn:
        # Avoid server-side timeouts during multi-hour bulk loads.
        conn.execute("SET statement_timeout = 0")
        try:
            conn.execute("SET idle_in_transaction_session_timeout = 0")
        except psycopg.Error as e:
            logging.warning("idle_in_transaction_session_timeout unset: %s", e)
        conn.commit()

        if not args.skip_features:
            load_releases_features(
                fs_path,
                conn,
                log_every=args.log_every,
                copy_chunk_rows=args.copy_chunk_rows,
            )
        else:
            logging.info("Skipping releases_features (--skip-features)")

        if not args.skip_marketplace:
            load_marketplace_stats(
                mp_path,
                conn,
                batch_size=max(1, args.batch_size),
                log_every=args.log_every,
            )
        else:
            logging.info("Skipping marketplace_stats (--skip-marketplace)")

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
