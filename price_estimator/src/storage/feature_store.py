"""SQLite feature store: catalog-derived features per release_id."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator

# Column order for INSERT/UPSERT (must match table). Plan §1b: no FS community
# columns — use ``marketplace_stats.community_want`` / ``community_have`` at training time.
RELEASES_FEATURES_COLUMNS: list[str] = [
    "release_id",
    "master_id",
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

MASTERS_FEATURES_COLUMNS: list[str] = [
    "master_id",
    "main_release_id",
    "title",
    "year",
    "primary_artist_id",
    "primary_artist_name",
    "primary_genre",
    "primary_style",
    "artists_json",
    "genres_json",
    "styles_json",
    "data_quality",
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

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {str(r[1]) for r in cur.fetchall()}

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS releases_features (
                    release_id TEXT PRIMARY KEY,
                    master_id TEXT,
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
            self._migrate_drop_fs_community_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS masters_features (
                    master_id TEXT PRIMARY KEY,
                    main_release_id TEXT,
                    title TEXT,
                    year INTEGER,
                    primary_artist_id TEXT,
                    primary_artist_name TEXT,
                    primary_genre TEXT,
                    primary_style TEXT,
                    artists_json TEXT,
                    genres_json TEXT,
                    styles_json TEXT,
                    data_quality TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_masters_features_artist_title_year
                ON masters_features(primary_artist_name, title, year)
                """
            )
            conn.commit()

    def _migrate_drop_fs_community_columns(self, conn: sqlite3.Connection) -> None:
        """Plan §1b: drop legacy ``want_count`` / ``have_count`` / ``want_have_ratio``."""
        while True:
            existing = self._existing_columns(conn)
            drops = [
                c for c in ("want_count", "have_count", "want_have_ratio") if c in existing
            ]
            if not drops:
                return
            col = drops[0]
            try:
                conn.execute(f"ALTER TABLE releases_features DROP COLUMN {col}")
            except sqlite3.OperationalError:
                self._rebuild_releases_features_without_community(conn)
                return

    def _rebuild_releases_features_without_community(self, conn: sqlite3.Connection) -> None:
        old_name = "releases_features_old"
        conn.execute(f"ALTER TABLE releases_features RENAME TO {old_name}")
        conn.execute(
            """
            CREATE TABLE releases_features (
                release_id TEXT PRIMARY KEY,
                master_id TEXT,
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
        old = self._table_columns(conn, old_name)
        parts = [c if c in old else "NULL" for c in RELEASES_FEATURES_COLUMNS]
        cols_csv = ", ".join(RELEASES_FEATURES_COLUMNS)
        sel_csv = ", ".join(parts)
        conn.execute(
            f"INSERT INTO releases_features ({cols_csv}) "
            f"SELECT {sel_csv} FROM {old_name}"
        )
        conn.execute(f"DROP TABLE {old_name}")

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

    def upsert_many_masters(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update many master rows (Discogs masters dump ingest)."""
        cols = MASTERS_FEATURES_COLUMNS
        placeholders = ",".join("?" * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "master_id")
        sql = f"""
                INSERT INTO masters_features ({",".join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(master_id) DO UPDATE SET {updates}
                """
        n = 0
        with self._connect() as conn:
            for row in rows:
                mid = str(row.get("master_id", "")).strip()
                if not mid:
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

    def ping(self) -> None:
        """Cheap DB connectivity check for probes (avoid COUNT on huge tables)."""
        with self._connect() as conn:
            conn.execute("SELECT 1")

    def iter_release_ids(
        self,
        *,
        sort_by: str = "release_id",
        min_have: int = 0,
        min_want: int = 0,
        marketplace_db_path: Path | str | None = None,
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
          ``have_count`` — descending ``marketplace_stats.community_have``, then
          ``release_id`` (requires ``marketplace_db_path``);
          ``want_count`` — descending ``community_want`` (requires ``marketplace_db_path``);
          ``popularity`` — descending ``community_have + community_want`` (requires
          ``marketplace_db_path``);
          ``catalog_proxy`` — master + primary-artist catalog mass (see
          ``price_estimator.src.catalog_proxy``); ignores ``min_have`` / ``min_want``.
        ``min_have`` / ``min_want``: optional lower bounds on MP community counts
        (0 = no filter); not applied for ``catalog_proxy`` / ``release_id``.
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

        if sort_by in ("have_count", "want_count", "popularity"):
            if not marketplace_db_path:
                raise ValueError(
                    f"sort_by={sort_by!r} requires marketplace_db_path (plan §1b: "
                    "community counts live in marketplace_stats, not releases_features)."
                )

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
        ch = "COALESCE(m.community_have, 0)"
        cw = "COALESCE(m.community_want, 0)"
        if mh > 0:
            clauses.append(f"{ch} >= ?")
            params.append(mh)
        if mw > 0:
            clauses.append(f"{cw} >= ?")
            params.append(mw)
        if exclude_various_artists:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_various_primary_artist,
            )

            clauses.append(sql_exclude_various_primary_artist("f.artists_json"))
        if exclude_file_formats:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_file_format_releases,
            )

            clauses.append(
                sql_exclude_file_format_releases("f.formats_json", "f.format_desc")
            )
        if exclude_unofficial_releases:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_unofficial_releases,
            )

            clauses.append(
                sql_exclude_unofficial_releases("f.formats_json", "f.format_desc")
            )
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""

        vinyl_k = None
        if prefer_vinyl_tiebreak:
            from price_estimator.src.catalog_proxy import sql_vinyl_format_rank

            vinyl_k = sql_vinyl_format_rank("f.formats_json", "f.format_desc")

        if sort_by == "have_count":
            if vinyl_k:
                order = f"ORDER BY {ch} DESC, {vinyl_k} DESC, f.release_id ASC"
            else:
                order = f"ORDER BY {ch} DESC, f.release_id ASC"
        elif sort_by == "want_count":
            if vinyl_k:
                order = f"ORDER BY {cw} DESC, {vinyl_k} DESC, f.release_id ASC"
            else:
                order = f"ORDER BY {cw} DESC, f.release_id ASC"
        elif sort_by == "popularity":
            pop = f"({ch} + {cw})"
            if vinyl_k:
                order = f"ORDER BY {pop} DESC, {vinyl_k} DESC, f.release_id ASC"
            else:
                order = f"ORDER BY {pop} DESC, f.release_id ASC"
        else:
            order = "ORDER BY f.release_id ASC"

        if sort_by == "release_id":
            sql = f"SELECT f.release_id FROM releases_features AS f {where}{order}"
            with self._connect() as conn:
                cur = conn.execute(sql, tuple(params))
                for row in cur:
                    yield str(row[0])
            return

        mp = str(Path(marketplace_db_path).resolve())  # type: ignore[arg-type]
        with self._connect() as conn:
            conn.execute("ATTACH DATABASE ? AS mdb", (mp,))
            try:
                sql = (
                    "SELECT f.release_id FROM releases_features AS f "
                    "LEFT JOIN mdb.marketplace_stats AS m ON m.release_id = f.release_id "
                    f"{where}{order}"
                )
                cur = conn.execute(sql, tuple(params))
                for row in cur:
                    yield str(row[0])
            finally:
                conn.execute("DETACH DATABASE mdb")

    def iter_community_release_rows(
        self,
        *,
        sort_by: str,
        marketplace_db_path: Path | str,
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
        ``marketplace_db_path``: path to ``marketplace_stats.sqlite`` (required).
        Honors ``exclude_unofficial_releases`` like ``iter_release_ids``.
        """
        valid = ("have_count", "want_count", "popularity")
        if sort_by not in valid:
            raise ValueError(f"sort_by must be one of {valid}, got {sort_by!r}")

        clauses: list[str] = []
        params: list[int] = []
        mh = max(0, int(min_have))
        mw = max(0, int(min_want))
        ch = "COALESCE(m.community_have, 0)"
        cw = "COALESCE(m.community_want, 0)"
        if mh > 0:
            clauses.append(f"{ch} >= ?")
            params.append(mh)
        if mw > 0:
            clauses.append(f"{cw} >= ?")
            params.append(mw)
        if exclude_various_artists:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_various_primary_artist,
            )

            clauses.append(sql_exclude_various_primary_artist("f.artists_json"))
        if exclude_file_formats:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_file_format_releases,
            )

            clauses.append(
                sql_exclude_file_format_releases("f.formats_json", "f.format_desc")
            )
        if exclude_unofficial_releases:
            from price_estimator.src.catalog_proxy import (
                sql_exclude_unofficial_releases,
            )

            clauses.append(
                sql_exclude_unofficial_releases("f.formats_json", "f.format_desc")
            )
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""

        vinyl_k = None
        if prefer_vinyl_tiebreak:
            from price_estimator.src.catalog_proxy import sql_vinyl_format_rank

            vinyl_k = sql_vinyl_format_rank("f.formats_json", "f.format_desc")

        if sort_by == "have_count":
            if vinyl_k:
                order = f"ORDER BY {ch} DESC, {vinyl_k} DESC, f.release_id ASC"
            else:
                order = f"ORDER BY {ch} DESC, f.release_id ASC"
        elif sort_by == "want_count":
            if vinyl_k:
                order = f"ORDER BY {cw} DESC, {vinyl_k} DESC, f.release_id ASC"
            else:
                order = f"ORDER BY {cw} DESC, f.release_id ASC"
        else:
            pop = f"({ch} + {cw})"
            if vinyl_k:
                order = f"ORDER BY {pop} DESC, {vinyl_k} DESC, f.release_id ASC"
            else:
                order = f"ORDER BY {pop} DESC, f.release_id ASC"

        vk_sel = vinyl_k if vinyl_k else "0"
        mp = str(Path(marketplace_db_path).resolve())
        with self._connect() as conn:
            conn.execute("ATTACH DATABASE ? AS mdb", (mp,))
            try:
                sql = (
                    f"SELECT f.release_id, CAST(({vk_sel}) AS INTEGER) AS _v "
                    "FROM releases_features AS f "
                    "LEFT JOIN mdb.marketplace_stats AS m ON m.release_id = f.release_id "
                    f"{where}{order}"
                )
                cur = conn.execute(sql, tuple(params))
                for row in cur:
                    yield str(row[0]), int(row[1] or 0)
            finally:
                conn.execute("DETACH DATABASE mdb")
