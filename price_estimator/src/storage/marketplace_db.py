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


def extract_release_listing_fields(release: dict[str, Any]) -> dict[str, Any]:
    """
    Scalars from ``GET /releases/{id}`` used for training/features.

    ``lowest_price`` may be a number or a money object; ``num_for_sale`` and
    ``community.want`` / ``community.have`` are documented on the Release resource.
    """
    comm = release.get("community") if isinstance(release.get("community"), dict) else {}
    want = comm.get("want")
    have = comm.get("have")
    try:
        cw = int(want) if want is not None else 0
    except (TypeError, ValueError):
        cw = 0
    try:
        ch = int(have) if have is not None else 0
    except (TypeError, ValueError):
        ch = 0
    lp_raw = release.get("lowest_price")
    lp = _parse_price_field(lp_raw if isinstance(lp_raw, dict) else lp_raw)
    nfs = release.get("num_for_sale")
    try:
        num_sale = int(nfs) if nfs is not None else 0
    except (TypeError, ValueError):
        num_sale = 0
    return {
        "release_lowest_price": lp,
        "release_num_for_sale": num_sale,
        "community_want": cw,
        "community_have": ch,
    }


def price_suggestions_ladder_from_json(raw_json: str | None) -> dict[str, dict[str, Any]]:
    """
    Parse stored ``GET /marketplace/price_suggestions/{id}`` JSON: one entry per
    Discogs media grade (e.g. ``"Near Mint (NM or M-)"`` → ``{"value", "currency"}``).

    The collector persists the **full** API object in ``price_suggestions_json``;
    use this (or ``price_suggestion_values_by_grade``) when fitting condition rules
    from the whole ladder, not only the training ``price_suggestion_grade``.
    """
    if raw_json is None or not str(raw_json).strip():
        return {}
    try:
        d = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(d, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in d.items():
        if isinstance(v, dict) and ("value" in v or "currency" in v):
            out[str(k)] = dict(v)
    return out


def price_suggestion_values_by_grade(raw_json: str | None) -> dict[str, float]:
    """Grade → positive ``value`` for each condition (numeric extraction only)."""
    ladder = price_suggestions_ladder_from_json(raw_json)
    out: dict[str, float] = {}
    for grade, entry in ladder.items():
        x = _parse_price_field(entry)
        if x is not None and x > 0:
            out[grade] = float(x)
    return out


def merge_release_listing_into_norm(
    norm: dict[str, Any],
    stats_payload: dict[str, Any],
    release_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    When ``GET /marketplace/stats`` is not used, fill ``lowest_price`` /
    ``median_price`` / ``num_for_sale`` from ``GET /releases`` for the same
    fields the stats endpoint would have covered.

    If the stats payload already includes a lowest price or ``num_for_sale``,
    those values are kept.
    """
    if not isinstance(release_payload, dict):
        return norm
    ext = extract_release_listing_fields(release_payload)
    lp = ext.get("release_lowest_price")
    try:
        rns = int(ext.get("release_num_for_sale") or 0)
    except (TypeError, ValueError):
        rns = 0

    sp = stats_payload if isinstance(stats_payload, dict) else {}
    stats_has_lowest = sp.get("lowest_price") is not None or sp.get("lowest") is not None
    stats_has_nfs = sp.get("num_for_sale") is not None or sp.get(
        "for_sale_count"
    ) is not None

    if not stats_has_lowest and lp is not None:
        norm["lowest_price"] = lp
        if norm.get("median_price") is None:
            norm["median_price"] = lp

    if not stats_has_nfs:
        norm["num_for_sale"] = rns

    return norm


def normalize_marketplace_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize Discogs marketplace stats JSON to scalars.

    Field names vary; we map common keys. Full payload is still stored in
    ``raw_json`` on upsert.

    Note: Discogs often omits a true ``median``; ``median_price`` in the return
    value may equal ``lowest_price`` after fallback (see ``median`` assignment
    below). Prefer ``release_lowest_price`` from ``GET /releases`` when present.
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
    ("release_raw_json", "TEXT"),
    ("price_suggestions_json", "TEXT"),
    ("release_lowest_price", "REAL"),
    ("release_num_for_sale", "INTEGER"),
    ("community_want", "INTEGER"),
    ("community_have", "INTEGER"),
]


class MarketplaceStatsDB:
    """SQLite cache for per-release marketplace listing data and training labels."""

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

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = {k: row[k] for k in row.keys()}
        return d

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
        """
        Upsert marketplace row. Legacy call: ``upsert(rid, stats_dict)`` only
        updates stats + ``raw_json``. Optional ``release_payload`` and
        ``price_suggestions_payload`` fill extended columns; omitted parts keep
        previous values when updating an existing row.

        When the stats payload omits listing fields, ``release_payload`` is used
        to fill ``lowest_price`` / ``median_price`` / ``num_for_sale`` (same as
        ``full`` collector: ``upsert(rid, {}, release_payload=...)``).

        When ``price_suggestions_payload`` is ``{}`` but the row already has a
        non-empty ``price_suggestions_json``, the previous JSON is kept so the
        full per-condition ladder is not lost on transient API gaps.
        """
        rid = str(release_id).strip()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM marketplace_stats WHERE release_id = ?",
                (rid,),
            )
            prev_row = cur.fetchone()
            prev = self._row_to_dict(prev_row) if prev_row else {}

            norm = normalize_marketplace_stats(payload)
            norm = merge_release_listing_into_norm(norm, payload, release_payload)
            stats_body = raw_json if raw_json is not None else json.dumps(payload)

            rel_raw = release_raw_json
            if release_payload is not None and rel_raw is None:
                rel_raw = json.dumps(release_payload, ensure_ascii=False)
            rel_ext = (
                extract_release_listing_fields(release_payload)
                if isinstance(release_payload, dict)
                else {}
            )

            ps_raw = price_suggestions_json
            if price_suggestions_payload is not None and ps_raw is None:
                new_ps = (
                    price_suggestions_payload
                    if isinstance(price_suggestions_payload, dict)
                    else {}
                )
                # Discogs returns {} when seller settings are incomplete; do not wipe a
                # previously stored full per-condition ladder (used for condition rules).
                if len(new_ps) == 0:
                    prev_psj = prev.get("price_suggestions_json")
                    if prev_psj and str(prev_psj).strip() not in ("", "{}"):
                        try:
                            prev_obj = json.loads(prev_psj)
                            if isinstance(prev_obj, dict) and len(prev_obj) > 0:
                                ps_raw = prev_psj
                        except json.JSONDecodeError:
                            pass
                if ps_raw is None:
                    ps_raw = json.dumps(
                        new_ps, ensure_ascii=False, separators=(",", ":")
                    )

            def _coalesce(new: Any, old_key: str) -> Any:
                if new is not None:
                    return new
                return prev.get(old_key)

            rlj = rel_raw if rel_raw is not None else prev.get("release_raw_json")
            psj = ps_raw if ps_raw is not None else prev.get("price_suggestions_json")

            rlp = rel_ext.get("release_lowest_price")
            rns = rel_ext.get("release_num_for_sale")
            cw = rel_ext.get("community_want")
            ch = rel_ext.get("community_have")
            if release_payload is None:
                rlp = _coalesce(None, "release_lowest_price")
                rns = _coalesce(None, "release_num_for_sale")
                cw = _coalesce(None, "community_want")
                ch = _coalesce(None, "community_have")
            else:
                rlp = rlp if rlp is not None else prev.get("release_lowest_price")
                rns = rns if rns is not None else prev.get("release_num_for_sale")
                cw = cw if cw is not None else prev.get("community_want")
                ch = ch if ch is not None else prev.get("community_have")

            conn.execute(
                """
                INSERT INTO marketplace_stats (
                    release_id, fetched_at, lowest_price, median_price,
                    highest_price, num_for_sale, blocked_from_sale, raw_json,
                    release_raw_json, price_suggestions_json,
                    release_lowest_price, release_num_for_sale,
                    community_want, community_have
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_id) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    lowest_price = excluded.lowest_price,
                    median_price = excluded.median_price,
                    highest_price = excluded.highest_price,
                    num_for_sale = excluded.num_for_sale,
                    blocked_from_sale = excluded.blocked_from_sale,
                    raw_json = excluded.raw_json,
                    release_raw_json = excluded.release_raw_json,
                    price_suggestions_json = excluded.price_suggestions_json,
                    release_lowest_price = excluded.release_lowest_price,
                    release_num_for_sale = excluded.release_num_for_sale,
                    community_want = excluded.community_want,
                    community_have = excluded.community_have
                """,
                (
                    rid,
                    now,
                    norm["lowest_price"],
                    norm["median_price"],
                    norm["highest_price"],
                    norm["num_for_sale"],
                    norm["blocked_from_sale"],
                    stats_body,
                    rlj,
                    psj,
                    rlp,
                    rns,
                    cw,
                    ch,
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
        for k in (
            "release_raw_json",
            "price_suggestions_json",
            "release_lowest_price",
            "release_num_for_sale",
            "community_want",
            "community_have",
        ):
            if k in keys:
                out[k] = row[k]
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
