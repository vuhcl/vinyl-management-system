"""Shared SQLite connection policy for price_estimator storage modules."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def open_sqlite(
    path: Path | str,
    *,
    wal: bool = True,
    busy_timeout_ms: int = 30_000,
    timeout: float = 30.0,
    row_factory: Any = sqlite3.Row,
) -> sqlite3.Connection:
    """
    Open a SQLite connection with row factory, optional WAL, and busy timeout.

    Matches ``MarketplaceStatsDB`` / ``SaleHistoryDB`` defaults; use ``wal=False`` only
    for read-only probes that must not flip journal mode.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    if row_factory is not None:
        conn.row_factory = row_factory
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    return conn
