"""Export ``releases_features`` from SQLite to canonical Parquet (Cloud-SQL–aligned order).

Column order matches ``RELEASES_FEATURES_COLUMNS`` in
``price_estimator.src.storage.feature_store`` so the same file can be diffed
against Postgres/Cloud SQL exports.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from price_estimator.src.storage.feature_store import RELEASES_FEATURES_COLUMNS


def export_releases_features_to_parquet(
    sqlite_path: Path | str,
    parquet_path: Path | str,
    *,
    table: str = "releases_features",
) -> int:
    """Read ``table`` from SQLite, reorder columns, write Parquet.

    Returns the number of rows written.
    """
    src = Path(sqlite_path)
    out = Path(parquet_path)
    if not src.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(src))
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        have = {str(r[1]) for r in cur.fetchall()}
    finally:
        conn.close()
    missing = [c for c in RELEASES_FEATURES_COLUMNS if c not in have]
    if missing:
        raise ValueError(
            f"Table {table!r} missing columns {missing}; cannot export canonical Parquet"
        )
    select_cols = ", ".join(RELEASES_FEATURES_COLUMNS)
    conn = sqlite3.connect(str(src))
    try:
        df = pd.read_sql_query(f"SELECT {select_cols} FROM {table}", conn)
    finally:
        conn.close()
    for col in df.columns:
        if col in (
            "is_original_pressing",
            "is_colored_vinyl",
            "is_picture_disc",
            "is_promo",
        ):
            df[col] = df[col].astype("Int64")
        elif col in ("decade", "year", "label_tier"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df.to_parquet(out, index=False)
    return int(len(df))
