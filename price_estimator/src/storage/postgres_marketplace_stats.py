"""PostgreSQL marketplace_stats (parity with SQLite ``MarketplaceStatsDB``)."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .marketplace_db import compute_marketplace_upsert_values
from .marketplace_projection import marketplace_stats_public_dict


class PostgresMarketplaceStats:
    """Marketplace listing cache + labels for VinylIQ inference."""

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
                    "SELECT * FROM marketplace_stats WHERE release_id = %s",
                    (rid,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return marketplace_stats_public_dict(dict(row))

    def upsert(
        self,
        release_id: str,
        payload: dict[str, Any],
        *,
        raw_json: str | None = None,
        release_payload: dict[str, Any] | None = None,
        release_raw_json: str | None = None,
        price_suggestions_payload: dict[str, Any] | None = None,
        price_suggestions_json: str | None = None,
    ) -> dict[str, Any]:
        rid = str(release_id).strip()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM marketplace_stats WHERE release_id = %s",
                    (rid,),
                )
                prev_row = cur.fetchone()
            prev = dict(prev_row) if prev_row else {}
            comp = compute_marketplace_upsert_values(
                rid,
                payload,
                prev,
                raw_json=raw_json,
                release_payload=release_payload,
                release_raw_json=release_raw_json,
                price_suggestions_payload=price_suggestions_payload,
                price_suggestions_json=price_suggestions_json,
            )
            norm = comp["norm"]
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO marketplace_stats (
                        release_id, fetched_at, num_for_sale,
                        blocked_from_sale,
                        raw_json, release_raw_json,
                        price_suggestions_json,
                        release_lowest_price, release_num_for_sale,
                        community_want, community_have
                    ) VALUES (
                        %(release_id)s, %(fetched_at)s,
                        %(num_for_sale)s,
                        %(blocked_from_sale)s, %(raw_json)s,
                        %(release_raw_json)s,
                        %(price_suggestions_json)s,
                        %(release_lowest_price)s,
                        %(release_num_for_sale)s,
                        %(community_want)s,
                        %(community_have)s
                    )
                    ON CONFLICT (release_id) DO UPDATE SET
                        fetched_at = EXCLUDED.fetched_at,
                        num_for_sale = EXCLUDED.num_for_sale,
                        blocked_from_sale = EXCLUDED.blocked_from_sale,
                        raw_json = EXCLUDED.raw_json,
                        release_raw_json = EXCLUDED.release_raw_json,
                        price_suggestions_json =
                            EXCLUDED.price_suggestions_json,
                        release_lowest_price =
                            EXCLUDED.release_lowest_price,
                        release_num_for_sale =
                            EXCLUDED.release_num_for_sale,
                        community_want = EXCLUDED.community_want,
                        community_have = EXCLUDED.community_have
                    """,
                    {
                        "release_id": comp["release_id"],
                        "fetched_at": comp["fetched_at"],
                        "num_for_sale": comp["num_for_sale"],
                        "blocked_from_sale": comp["blocked_from_sale"],
                        "raw_json": comp["raw_json"],
                        "release_raw_json": comp["release_raw_json"],
                        "price_suggestions_json": comp["price_suggestions_json"],
                        "release_lowest_price": comp["release_lowest_price"],
                        "release_num_for_sale": comp["release_num_for_sale"],
                        "community_want": comp["community_want"],
                        "community_have": comp["community_have"],
                    },
                )
        return norm
