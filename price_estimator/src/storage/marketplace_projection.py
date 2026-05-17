"""Inference-facing projection of marketplace_stats rows (SQLite + Postgres parity)."""
from __future__ import annotations

from typing import Any, Mapping

_OPTIONAL_MARKETPLACE_KEYS: tuple[str, ...] = (
    "release_raw_json",
    "price_suggestions_json",
    "release_lowest_price",
    "release_num_for_sale",
    "community_want",
    "community_have",
)


def marketplace_stats_public_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    """
    Map a full ``marketplace_stats`` row to the cache dict used at inference / training join.

    Base keys are always present; ``blocked_from_sale`` and optional release/community
    fields are included when the row provides them.

    Note: ``k in sqlite3.Row`` tests **values**, not column names — use ``row.keys()``.
    """
    colnames = set(row.keys())
    out: dict[str, Any] = {
        "release_id": row["release_id"],
        "fetched_at": row["fetched_at"],
        "num_for_sale": row["num_for_sale"],
        "raw_json": row["raw_json"],
    }
    if "blocked_from_sale" in colnames:
        out["blocked_from_sale"] = row["blocked_from_sale"]
    for k in _OPTIONAL_MARKETPLACE_KEYS:
        if k in colnames:
            out[k] = row[k]
    return out
