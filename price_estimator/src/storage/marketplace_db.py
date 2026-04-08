"""SQLite cache and label store for Discogs marketplace/stats responses."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_price_field(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        x = v.get("value")
        if x is not None:
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _blocked_from_sale_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return 1 if int(v) else 0
    except (TypeError, ValueError):
        return None


def normalize_marketplace_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize Discogs marketplace stats JSON to scalars.

    Field names vary; we map common keys. Full payload is still stored in
    ``raw_json`` on upsert.
    """
    lowest = _parse_price_field(
        payload.get("lowest_price") or payload.get("lowest")
    )
    median = _parse_price_field(
        payload.get("median_price")
        or payload.get("median")
        or payload.get("blocked_lowest_price")
    )
    if median is None:
        median = lowest
    highest = _parse_price_field(payload.get("highest_price"))
    nfs = payload.get("num_for_sale")
    if nfs is None:
        nfs = payload.get("for_sale_count")
    try:
        num_for_sale = int(nfs) if nfs is not None else 0
    except (TypeError, ValueError):
        num_for_sale = 0
    blocked = _blocked_from_sale_int(payload.get("blocked_from_sale"))
    return {
        "lowest_price": lowest,
        "median_price": median,
        "highest_price": highest,
        "num_for_sale": num_for_sale,
        "blocked_from_sale": blocked,
    }


_MARKETPLACE_MIGRATIONS: list[tuple[str, str]] = [
    ("highest_price", "REAL"),
    ("blocked_from_sale", "INTEGER"),
]


class MarketplaceStatsDB:
    """Persistent cache + training labels for GET /marketplace/stats/{release_id}."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _existing_columns(self, conn: sqlite3.Connection) -> set[str]:
        cur = conn.execute("PRAGMA table_info(marketplace_stats)")
        return {str(r[1]) for r in cur.fetchall()}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS marketplace_stats (
                    release_id TEXT PRIMARY KEY,
                    fetched_at TEXT NOT NULL,
                    lowest_price REAL,
                    median_price REAL,
                    num_for_sale INTEGER,
                    raw_json TEXT NOT NULL
                )
                """
            )
            existing = self._existing_columns(conn)
            for col, sql_type in _MARKETPLACE_MIGRATIONS:
                if col not in existing:
                    conn.execute(
                        f"ALTER TABLE marketplace_stats ADD COLUMN {col} {sql_type}"
                    )
            conn.commit()

    def upsert(
        self,
        release_id: str,
        payload: dict[str, Any],
        *,
        raw_json: str | None = None,
    ) -> dict[str, Any]:
        rid = str(release_id).strip()
        norm = normalize_marketplace_stats(payload)
        body = raw_json if raw_json is not None else json.dumps(payload)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO marketplace_stats (
                    release_id, fetched_at, lowest_price, median_price,
                    highest_price, num_for_sale, blocked_from_sale, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_id) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    lowest_price = excluded.lowest_price,
                    median_price = excluded.median_price,
                    highest_price = excluded.highest_price,
                    num_for_sale = excluded.num_for_sale,
                    blocked_from_sale = excluded.blocked_from_sale,
                    raw_json = excluded.raw_json
                """,
                (
                    rid,
                    now,
                    norm["lowest_price"],
                    norm["median_price"],
                    norm["highest_price"],
                    norm["num_for_sale"],
                    norm["blocked_from_sale"],
                    body,
                ),
            )
            conn.commit()
        return norm

    def get(self, release_id: str) -> dict[str, Any] | None:
        rid = str(release_id).strip()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM marketplace_stats WHERE release_id = ?",
                (rid,),
            )
            row = cur.fetchone()
        if not row:
            return None
        out = {
            "release_id": row["release_id"],
            "fetched_at": row["fetched_at"],
            "lowest_price": row["lowest_price"],
            "median_price": row["median_price"],
            "num_for_sale": row["num_for_sale"],
            "raw_json": row["raw_json"],
        }
        keys = row.keys()
        if "highest_price" in keys:
            out["highest_price"] = row["highest_price"]
        if "blocked_from_sale" in keys:
            out["blocked_from_sale"] = row["blocked_from_sale"]
        return out

    def iter_release_ids(self) -> list[str]:
        with self._connect() as conn:
            cur = conn.execute("SELECT release_id FROM marketplace_stats")
            return [r[0] for r in cur.fetchall()]

    def existing_release_ids(self) -> set[str]:
        """All release_id keys present (for resume / skip-already-fetched)."""
        with self._connect() as conn:
            cur = conn.execute("SELECT release_id FROM marketplace_stats")
            return {str(r[0]) for r in cur.fetchall()}

    def count_rows(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM marketplace_stats")
            return int(cur.fetchone()[0])

    def has_release_id(self, release_id: str) -> bool:
        """Cheap existence check for resume without loading all keys into memory."""
        rid = str(release_id).strip()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM marketplace_stats WHERE release_id = ? LIMIT 1",
                (rid,),
            )
            return cur.fetchone() is not None
