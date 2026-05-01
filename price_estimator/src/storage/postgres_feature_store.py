"""PostgreSQL feature store for inference-time ``releases_features`` reads."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class PostgresFeatureStore:
    """Read-only ``releases_features`` backed by Cloud SQL / Postgres."""

    def __init__(self, dsn: str) -> None:
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=5,
            open=True,
        )

    def get(self, release_id: str) -> dict[str, Any] | None:
        rid = str(release_id).strip()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM releases_features WHERE release_id = %s",
                    (rid,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM releases_features")
                out = cur.fetchone()
                return int(out[0]) if out else 0
