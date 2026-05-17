"""SQLite store for Discogs web-scraped per-release sale history."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .sqlite_util import open_sqlite
from price_estimator.src.scrape.discogs_sale_history_parse import (
    ParsedSaleHistory,
    utc_now_iso,
)


class SaleHistoryDB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return open_sqlite(self.path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS release_sale_summary (
                    release_id TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    last_sold_on TEXT,
                    average REAL,
                    median REAL,
                    high REAL,
                    low REAL,
                    raw_json TEXT NOT NULL,
                    PRIMARY KEY (release_id, fetched_at)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS release_sale (
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
                CREATE TABLE IF NOT EXISTS sale_history_fetch_status (
                    release_id TEXT PRIMARY KEY,
                    fetched_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    num_rows INTEGER NOT NULL DEFAULT 0,
                    parse_warnings TEXT
                )
                """
            )
            conn.commit()

    def upsert_parsed(
        self,
        parsed: ParsedSaleHistory,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        rid = parsed.release_id
        now = utc_now_iso()
        summary = parsed.summary
        raw_summary = json.dumps(
            asdict(summary) if summary else {},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._connect() as conn:
            if summary:
                conn.execute(
                    """
                    INSERT INTO release_sale_summary (
                        release_id, fetched_at, last_sold_on, average, median,
                        high, low, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rid,
                        now,
                        summary.last_sold_on,
                        summary.average,
                        summary.median,
                        summary.high,
                        summary.low,
                        raw_summary,
                    ),
                )
            for row in parsed.rows:
                h = row.row_hash(rid)
                conn.execute(
                    """
                    INSERT INTO release_sale (
                        release_id, row_hash, order_date, media_condition,
                        sleeve_condition, price_original_text,
                        price_user_currency_text, seller_comments, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(release_id, row_hash) DO UPDATE SET
                        order_date = excluded.order_date,
                        media_condition = excluded.media_condition,
                        sleeve_condition = excluded.sleeve_condition,
                        price_original_text = excluded.price_original_text,
                        price_user_currency_text = excluded.price_user_currency_text,
                        seller_comments = excluded.seller_comments,
                        fetched_at = excluded.fetched_at
                    """,
                    (
                        rid,
                        h,
                        row.order_date,
                        row.media_condition,
                        row.sleeve_condition,
                        row.price_original_text,
                        row.price_user_currency_text,
                        row.seller_comments,
                        now,
                    ),
                )
            conn.execute(
                """
                INSERT INTO sale_history_fetch_status (
                    release_id, fetched_at, status, error, num_rows, parse_warnings
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_id) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    status = excluded.status,
                    error = excluded.error,
                    num_rows = excluded.num_rows,
                    parse_warnings = excluded.parse_warnings
                """,
                (
                    rid,
                    now,
                    status,
                    error,
                    len(parsed.rows),
                    json.dumps(parsed.parse_warnings, ensure_ascii=False),
                ),
            )
            conn.commit()

    def record_failure(
        self,
        release_id: str,
        error: str,
        *,
        warnings: list[str] | None = None,
    ) -> None:
        parsed = ParsedSaleHistory(
            release_id=str(release_id).strip(),
            summary=None,
            rows=[],
            parse_warnings=list(warnings or []) + ["fetch_error"],
        )
        self.upsert_parsed(parsed, status="error", error=error[:4000])

    def last_status(self, release_id: str) -> dict[str, Any] | None:
        rid = str(release_id).strip()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM sale_history_fetch_status WHERE release_id = ?",
                (rid,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {k: row[k] for k in row.keys()}

    def should_skip_resume(self, release_id: str, *, ok_hours: float) -> bool:
        st = self.last_status(release_id)
        if not st or st.get("status") != "ok":
            return False
        if ok_hours <= 0:
            return False
        try:
            prev = datetime.fromisoformat(
                str(st["fetched_at"]).replace("Z", "+00:00")
            )
        except ValueError:
            return False
        if prev.tzinfo is None:
            prev = prev.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ok_hours)
        return prev >= cutoff
