"""SQLite feature store: catalog-derived features per release_id."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

# Column order for INSERT/UPSERT (must match table).
RELEASES_FEATURES_COLUMNS: list[str] = [
    "release_id",
    "master_id",
    "want_count",
    "have_count",
    "want_have_ratio",
    "genre",
    "style",
    "decade",
    "year",
    "country",
    "label_tier",
    "is_original_pressing",
    "is_colored_vinyl",
    "is_picture_disc",
    "is_promo",
    "format_desc",
    "artists_json",
    "labels_json",
    "genres_json",
    "styles_json",
    "formats_json",
]

# (column_name, SQL type) appended via ALTER on existing DBs.
_RELEASES_FEATURES_MIGRATIONS: list[tuple[str, str]] = [
    ("artists_json", "TEXT"),
    ("labels_json", "TEXT"),
    ("genres_json", "TEXT"),
    ("styles_json", "TEXT"),
    ("formats_json", "TEXT"),
]


class FeatureStoreDB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _existing_columns(self, conn: sqlite3.Connection) -> set[str]:
        cur = conn.execute("PRAGMA table_info(releases_features)")
        return {str(r[1]) for r in cur.fetchall()}

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS releases_features (
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
                )
                """
            )
            existing = self._existing_columns(conn)
            for col, sql_type in _RELEASES_FEATURES_MIGRATIONS:
                if col not in existing:
                    conn.execute(
                        f"ALTER TABLE releases_features ADD COLUMN {col} {sql_type}"
                    )
            conn.commit()

    def upsert_row(self, row: dict[str, Any]) -> None:
        rid = str(row.get("release_id", "")).strip()
        if not rid:
            return
        cols = RELEASES_FEATURES_COLUMNS
        values = [row.get(c) for c in cols]
        placeholders = ",".join("?" * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "release_id")
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO releases_features ({",".join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(release_id) DO UPDATE SET {updates}
                """,
                values,
            )
            conn.commit()

    def upsert_many(self, rows: Iterable[dict[str, Any]]) -> int:
        """
        Insert or update many rows in one transaction (faster than repeated
        upsert_row for large dump ingests).
        """
        cols = RELEASES_FEATURES_COLUMNS
        placeholders = ",".join("?" * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "release_id")
        sql = f"""
                INSERT INTO releases_features ({",".join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(release_id) DO UPDATE SET {updates}
                """
        n = 0
        with self._connect() as conn:
            for row in rows:
                rid = str(row.get("release_id", "")).strip()
                if not rid:
                    continue
                values = [row.get(c) for c in cols]
                conn.execute(sql, values)
                n += 1
            conn.commit()
        return n

    def get(self, release_id: str) -> dict[str, Any] | None:
        rid = str(release_id).strip()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM releases_features WHERE release_id = ?",
                (rid,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def count(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM releases_features")
            return int(cur.fetchone()[0])

    def iter_release_ids(
        self,
        *,
        sort_by: str = "release_id",
        min_have: int = 0,
        min_want: int = 0,
        proxy_weight_master: float = 1.0,
        proxy_weight_artist: float = 1.0,
        exclude_various_artists: bool = False,
        exclude_file_formats: bool = False,
        exclude_unofficial_releases: bool = False,
        prefer_vinyl_tiebreak: bool = False,
    ) -> Iterator[str]:
        """
        Stream ``release_id`` values for export / stats collection.

        ``sort_by``:
          ``release_id`` — ascending ID;
          ``have_count`` — descending community have, then ``release_id``;
          ``want_count`` — descending community want, then ``release_id``;
          ``popularity`` — descending ``have_count + want_count``, then ``release_id``;
          ``catalog_proxy`` — master + primary-artist catalog mass (see
          ``price_estimator.src.catalog_proxy``); ignores ``min_have`` / ``min_want``.
        ``min_have`` / ``min_want``: optional lower bounds (0 = no filter); not
        applied for ``catalog_proxy``.
        ``exclude_various_artists``: if True, drop rows whose primary artist is
        Discogs Various (id 194), name contains ``various``, or name is
        **Unknown Artist** (not applied for ``catalog_proxy``; proxy SQL always
        excludes those rows).
        ``exclude_file_formats``: if True, drop Discogs **File** / digital rows
        (not applied for ``catalog_proxy``; proxy SQL always excludes them).
        ``exclude_unofficial_releases``: if True, drop rows tagged **Unofficial
        Release** in ``format_desc`` or format descriptions (not applied for
        ``catalog_proxy``; proxy SQL always excludes them).
        ``prefer_vinyl_tiebreak``: if True, break ties with vinyl **format rank**
        (LP before 12\"/10\"/7\") on community sorts (have / want / popularity only).
        """
        valid = (
            "release_id",
            "have_count",
            "want_count",
            "popularity",
            "catalog_proxy",
        )
        if sort_by not in valid:
            raise ValueError(f"sort_by must be one of {valid}, got {sort_by!r}")

        if sort_by == "catalog_proxy":
            from price_estimator.src.catalog_proxy import (
                sql_select_release_ids_ordered_by_catalog_proxy,
            )

            sql = sql_select_release_ids_ordered_by_catalog_proxy().strip()
            wm = float(proxy_weight_master)
            wa = float(proxy_weight_artist)
            with self._connect() as conn:
                cur = conn.execute(sql, (wm, wa))
                for row in cur:
                    yield str(row[0])
            return

        clauses: list[str] = []
        params: list[int] = []
        mh = max(0, int(min_have))
        mw = max(0, int(min_want))
        if mh > 0:
            clauses.append("COALESCE(have_count, 0) >= ?")
            params.append(mh)
        if mw > 0:
            clauses.append("COALESCE(want_count, 0) >= ?")
            params.append(mw)
        if exclude_various_artists:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_various_primary_artist,
            )

            clauses.append(sql_exclude_various_primary_artist("artists_json"))
        if exclude_file_formats:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_file_format_releases,
            )

            clauses.append(
                sql_exclude_file_format_releases("formats_json", "format_desc")
            )
        if exclude_unofficial_releases:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_unofficial_releases,
            )

            clauses.append(
                sql_exclude_unofficial_releases("formats_json", "format_desc")
            )
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""

        vinyl_k = None
        if prefer_vinyl_tiebreak:
            from price_estimator.src.catalog_proxy import sql_vinyl_format_rank

            vinyl_k = sql_vinyl_format_rank("formats_json", "format_desc")

        if sort_by == "have_count":
            if vinyl_k:
                order = (
                    f"ORDER BY have_count DESC, {vinyl_k} DESC, release_id ASC"
                )
            else:
                order = "ORDER BY have_count DESC, release_id ASC"
        elif sort_by == "want_count":
            if vinyl_k:
                order = (
                    f"ORDER BY want_count DESC, {vinyl_k} DESC, release_id ASC"
                )
            else:
                order = "ORDER BY want_count DESC, release_id ASC"
        elif sort_by == "popularity":
            if vinyl_k:
                order = (
                    "ORDER BY (COALESCE(have_count, 0) + COALESCE(want_count, 0)) "
                    f"DESC, {vinyl_k} DESC, release_id ASC"
                )
            else:
                order = (
                    "ORDER BY (COALESCE(have_count, 0) + COALESCE(want_count, 0)) "
                    "DESC, release_id ASC"
                )
        else:
            order = "ORDER BY release_id ASC"

        sql = f"SELECT release_id FROM releases_features {where}{order}"
        with self._connect() as conn:
            cur = conn.execute(sql, tuple(params))
            for row in cur:
                yield str(row[0])

    def iter_community_release_rows(
        self,
        *,
        sort_by: str,
        min_have: int = 0,
        min_want: int = 0,
        exclude_various_artists: bool = False,
        exclude_file_formats: bool = False,
        exclude_unofficial_releases: bool = False,
        prefer_vinyl_tiebreak: bool = False,
    ) -> Iterator[tuple[str, int]]:
        """
        Like ``iter_release_ids`` for community sorts, but yields
        ``(release_id, vinyl_rank)`` (0–4, LP highest) from catalog heuristics.

        ``sort_by``: ``have_count``, ``want_count``, or ``popularity`` only.
        Honors ``exclude_unofficial_releases`` like ``iter_release_ids``.
        """
        valid = ("have_count", "want_count", "popularity")
        if sort_by not in valid:
            raise ValueError(f"sort_by must be one of {valid}, got {sort_by!r}")

        clauses: list[str] = []
        params: list[int] = []
        mh = max(0, int(min_have))
        mw = max(0, int(min_want))
        if mh > 0:
            clauses.append("COALESCE(have_count, 0) >= ?")
            params.append(mh)
        if mw > 0:
            clauses.append("COALESCE(want_count, 0) >= ?")
            params.append(mw)
        if exclude_various_artists:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_various_primary_artist,
            )

            clauses.append(sql_exclude_various_primary_artist("artists_json"))
        if exclude_file_formats:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_file_format_releases,
            )

            clauses.append(
                sql_exclude_file_format_releases("formats_json", "format_desc")
            )
        if exclude_unofficial_releases:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_unofficial_releases,
            )

            clauses.append(
                sql_exclude_unofficial_releases("formats_json", "format_desc")
            )
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""

        vinyl_k = None
        if prefer_vinyl_tiebreak:
            from price_estimator.src.catalog_proxy import sql_vinyl_format_rank

            vinyl_k = sql_vinyl_format_rank("formats_json", "format_desc")

        if sort_by == "have_count":
            if vinyl_k:
                order = (
                    f"ORDER BY have_count DESC, {vinyl_k} DESC, release_id ASC"
                )
            else:
                order = "ORDER BY have_count DESC, release_id ASC"
        elif sort_by == "want_count":
            if vinyl_k:
                order = (
                    f"ORDER BY want_count DESC, {vinyl_k} DESC, release_id ASC"
                )
            else:
                order = "ORDER BY want_count DESC, release_id ASC"
        else:
            if vinyl_k:
                order = (
                    "ORDER BY (COALESCE(have_count, 0) + COALESCE(want_count, 0)) "
                    f"DESC, {vinyl_k} DESC, release_id ASC"
                )
            else:
                order = (
                    "ORDER BY (COALESCE(have_count, 0) + COALESCE(want_count, 0)) "
                    "DESC, release_id ASC"
                )

        vk_sel = vinyl_k if vinyl_k else "0"
        sql = (
            f"SELECT release_id, CAST(({vk_sel}) AS INTEGER) AS _v "
            f"FROM releases_features {where}{order}"
        )
        with self._connect() as conn:
            cur = conn.execute(sql, tuple(params))
            for row in cur:
                yield str(row[0]), int(row[1] or 0)
