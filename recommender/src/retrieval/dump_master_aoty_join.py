"""
Dump-join Discogs ``masters_features`` to AOTY ``albums`` for Phase 2a.

Uses ``releases_features`` to restrict to masters that have at least one
album-ish release (``Album`` / ``LP`` in ``format_desc`` or format
descriptions). Matches on normalized artist + title + year within ±1, with an
optional fuzzy title pass for the long tail.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# ``recommender.src.data`` may be editor-ignored; runtime imports still work.
from recommender.src.data.discogs_aoty_id_matching import (
    _normalize_text,
    _title_similarity,
)


# True when this *release row* looks like an album/LP listing (not single-only).
_ALBUMISH_RELEASE_SQL_PRED = """(
  INSTR(' ' || LOWER(COALESCE(rf.format_desc, '')) || ' ', ' album ') > 0
  OR INSTR(' ' || LOWER(COALESCE(rf.format_desc, '')) || ' ', ' lp ') > 0
  OR LOWER(TRIM(COALESCE(rf.format_desc, ''))) IN ('album', 'lp')
  OR (
    rf.formats_json IS NOT NULL
    AND TRIM(rf.formats_json) != ''
    AND TRIM(rf.formats_json) != '[]'
    AND EXISTS (
      SELECT 1 FROM json_each(rf.formats_json) AS fe
      WHERE EXISTS (
        SELECT 1 FROM json_each(
          CASE
            WHEN json_type(fe.value, '$.descriptions') = 'array'
            THEN json_extract(fe.value, '$.descriptions')
            ELSE '[]'
          END
        ) AS jd
        WHERE INSTR(' ' || LOWER(COALESCE(jd.value, '')) || ' ', ' album ') > 0
           OR INSTR(' ' || LOWER(COALESCE(jd.value, '')) || ' ', ' lp ') > 0
      )
    )
  )
)"""


def sql_distinct_albumish_master_ids() -> str:
    return f"""
        SELECT DISTINCT CAST(rf.master_id AS TEXT) AS master_id
        FROM releases_features AS rf
        WHERE rf.master_id IS NOT NULL
          AND TRIM(CAST(rf.master_id AS TEXT)) != ''
          AND ({_ALBUMISH_RELEASE_SQL_PRED})
    """


@dataclass(frozen=True)
class DumpJoinConfig:
    """Tuning for dump-join matching."""

    min_fuzzy_title_similarity: float = 0.88
    year_window: int = 1


def _norm(s: str | None) -> str:
    return _normalize_text((s or "").strip())


def fetch_albumish_master_ids(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(sql_distinct_albumish_master_ids())
    return {str(r[0]).strip() for r in cur.fetchall() if r[0] is not None}


def _create_temp_ids(conn: sqlite3.Connection, ids: Iterable[str]) -> None:
    conn.execute("DROP TABLE IF EXISTS _eligible_master")
    conn.execute("CREATE TEMP TABLE _eligible_master (master_id TEXT PRIMARY KEY)")
    conn.executemany(
        "INSERT OR IGNORE INTO _eligible_master (master_id) VALUES (?)",
        [(i,) for i in ids],
    )


def load_masters_for_join(
    feature_db: Path,
    eligible_master_ids: set[str],
) -> pd.DataFrame:
    """Load ``masters_features`` rows whose ``master_id`` is album-ish eligible."""
    if not eligible_master_ids:
        return pd.DataFrame()
    conn = sqlite3.connect(str(feature_db))
    try:
        _create_temp_ids(conn, eligible_master_ids)
        q = """
            SELECT m.master_id,
                   m.primary_artist_name,
                   m.title,
                   m.year
            FROM masters_features AS m
            INNER JOIN _eligible_master AS e ON e.master_id = m.master_id
        """
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()
    return df


def build_indexes(
    masters: pd.DataFrame,
) -> tuple[
    dict[tuple[str, int], list[tuple[str, str, int]]],
    dict[str, list[tuple[str, str, int]]],
]:
    """
    Return:
      ``by_artist_year[(norm_artist, year)] -> [(master_id, norm_title, raw_year), ...]``
      ``by_artist_only[norm_artist] -> [...]`` for masters with unknown year (0).
    """
    by_artist_year: dict[tuple[str, int], list[tuple[str, str, int]]] = defaultdict(
        list
    )
    by_artist_only: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row in masters.itertuples(index=False):
        mid = str(getattr(row, "master_id", "") or "").strip()
        if not mid:
            continue
        na = _norm(str(getattr(row, "primary_artist_name", "") or ""))
        nt = _norm(str(getattr(row, "title", "") or ""))
        try:
            y = int(getattr(row, "year", 0) or 0)
        except (TypeError, ValueError):
            y = 0
        if not na or not nt:
            continue
        tup = (mid, nt, y)
        if y <= 0:
            by_artist_only[na].append(tup)
        else:
            by_artist_year[(na, y)].append(tup)
    return dict(by_artist_year), dict(by_artist_only)


def _gather_candidates(
    na: str,
    album_y: int,
    by_artist_year: dict[tuple[str, int], list[tuple[str, str, int]]],
    by_artist_only: dict[str, list[tuple[str, str, int]]],
    year_window: int,
) -> list[tuple[str, str, int]]:
    seen: set[str] = set()
    out: list[tuple[str, str, int]] = []
    if album_y > 0:
        for dy in range(-year_window, year_window + 1):
            key = (na, album_y + dy)
            for tup in by_artist_year.get(key, ()):
                mid = tup[0]
                if mid not in seen:
                    seen.add(mid)
                    out.append(tup)
    for tup in by_artist_only.get(na, ()):
        mid = tup[0]
        if mid not in seen:
            seen.add(mid)
            out.append(tup)
    return out


def match_albums_dump_join(
    albums: pd.DataFrame,
    masters: pd.DataFrame,
    cfg: DumpJoinConfig,
    *,
    album_id_col: str = "album_id",
    artist_col: str = "artist",
    title_col: str = "album_title",
    year_col: str = "year",
) -> dict[str, str]:
    """
    Return ``discogs_master_id -> aoty_album_id`` for matches found.

    One album maps to at most one master (first exact title wins, else best
    fuzzy over threshold). One master maps to at most one album.
    """
    if albums.empty or masters.empty:
        return {}

    by_y, by_only = build_indexes(masters)
    used_masters: set[str] = set()
    out: dict[str, str] = {}

    if title_col not in albums.columns:
        alt = "title"
        if alt in albums.columns:
            title_col = alt
        else:
            raise ValueError(
                "albums must include 'album_title' or 'title' for dump-join."
            )

    for row in albums.itertuples(index=False):
        aid = str(getattr(row, album_id_col, "") or "").strip()
        if not aid:
            continue
        na = _norm(str(getattr(row, artist_col, "") or ""))
        nt = _norm(str(getattr(row, title_col, "") or ""))
        try:
            ay = int(getattr(row, year_col, 0) or 0)
        except (TypeError, ValueError):
            ay = 0
        if not na or not nt:
            continue

        cands = _gather_candidates(na, ay, by_y, by_only, cfg.year_window)
        cands = [t for t in cands if t[0] not in used_masters]
        if not cands:
            continue

        exact = [t for t in cands if t[1] == nt]
        pick: tuple[str, str, int] | None = None
        if len(exact) == 1:
            pick = exact[0]
        elif len(exact) > 1:
            exact.sort(key=lambda t: int(t[0]))
            pick = exact[0]
        else:
            best: tuple[str, str, float] | None = None
            for mid, mt, _yr in cands:
                sim = _title_similarity(nt, mt)
                if sim < cfg.min_fuzzy_title_similarity:
                    continue
                if best is None or sim > best[2] or (
                    sim == best[2] and mid < best[0]
                ):
                    best = (mid, mt, sim)
            if best is not None:
                bmid, bmt, _ = best
                pick = (bmid, bmt, 0)

        if pick is None:
            continue
        mid = pick[0]
        used_masters.add(mid)
        out[mid] = aid
    return out


def merge_master_to_aoty_maps(
    existing: dict[str, str],
    discovered: dict[str, str],
) -> dict[str, str]:
    """
    Merge *discovered* ``master_id -> album_id`` into *existing*.

    Never overwrites an existing ``master_id`` key. Skips a new pair when the
    ``album_id`` is already mapped to some (possibly other) master.
    """
    out = dict(existing)
    album_claimed = {str(v): str(k) for k, v in out.items()}

    for mid_s, aid_s in sorted(
        ((str(k).strip(), str(v).strip()) for k, v in discovered.items()),
        key=lambda kv: kv[0],
    ):
        if not mid_s or not aid_s:
            continue
        if mid_s in out:
            continue
        if aid_s in album_claimed:
            continue
        out[mid_s] = aid_s
        album_claimed[aid_s] = mid_s
    return out


def dump_join_stats(
    albums: pd.DataFrame,
    merged: dict[str, str],
    *,
    album_id_col: str = "album_id",
) -> dict[str, float | int]:
    n_albums = int(albums[album_id_col].astype(str).nunique())
    covered = 0
    aids = set(albums[album_id_col].astype(str).str.strip())
    inv = set(merged.values())
    for a in aids:
        if a in inv:
            covered += 1
    ratio = float(covered) / float(max(1, n_albums))
    return {
        "n_albums": n_albums,
        "n_albums_mapped": covered,
        "coverage": ratio,
        "n_master_keys": len(merged),
    }
