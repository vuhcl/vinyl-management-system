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
    When ``GET /marketplace/stats`` is not used, fill ``num_for_sale`` from
    ``GET /releases`` for the field the stats endpoint would have covered.
    Listing-floor data lives in ``release_lowest_price`` (written separately
    by ``upsert`` from ``extract_release_listing_fields``).

    If the stats payload already includes ``num_for_sale``, that value is kept.
    """
    if not isinstance(release_payload, dict):
        return norm
    ext = extract_release_listing_fields(release_payload)
    try:
        rns = int(ext.get("release_num_for_sale") or 0)
    except (TypeError, ValueError):
        rns = 0

    sp = stats_payload if isinstance(stats_payload, dict) else {}
    stats_has_nfs = sp.get("num_for_sale") is not None or sp.get(
        "for_sale_count"
    ) is not None

    if not stats_has_nfs:
        norm["num_for_sale"] = rns

    return norm


def normalize_marketplace_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize Discogs marketplace stats JSON to scalars.

    Full payload is still stored in ``raw_json`` on upsert. Listing-floor data
    lives in ``release_lowest_price`` (from ``GET /releases``); this helper
    only extracts the non-dollar fields kept on the table.
    """
    nfs = payload.get("num_for_sale")
    if nfs is None:
        nfs = payload.get("for_sale_count")
    try:
        num_for_sale = int(nfs) if nfs is not None else 0
    except (TypeError, ValueError):
        num_for_sale = 0
    blocked = _blocked_from_sale_int(payload.get("blocked_from_sale"))
    return {
        "num_for_sale": num_for_sale,
        "blocked_from_sale": blocked,
    }


def compute_marketplace_upsert_values(
    release_id: str,
    payload: dict[str, Any],
    prev: dict[str, Any],
    *,
    raw_json: str | None = None,
    release_payload: dict[str, Any] | None = None,
    release_raw_json: str | None = None,
    price_suggestions_payload: dict[str, Any] | None = None,
    price_suggestions_json: str | None = None,
    fetched_at: str | None = None,
) -> dict[str, Any]:
    """
    Pure merge/coalesce logic shared by SQLite ``MarketplaceStatsDB`` and
    ``PostgresMarketplaceStats``. ``prev`` is an empty dict on insert.

    Returns a dict with DB column keys plus ``norm`` (the return value contract
    of ``MarketplaceStatsDB.upsert``).
    """
    rid = str(release_id).strip()
    now = fetched_at or datetime.now(timezone.utc).isoformat()

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
            ps_raw = json.dumps(new_ps, ensure_ascii=False, separators=(",", ":"))

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

    return {
        "norm": norm,
        "release_id": rid,
        "fetched_at": now,
        "num_for_sale": norm["num_for_sale"],
        "blocked_from_sale": norm["blocked_from_sale"],
        "raw_json": stats_body,
        "release_raw_json": rlj,
        "price_suggestions_json": psj,
        "release_lowest_price": rlp,
        "release_num_for_sale": rns,
        "community_want": cw,
        "community_have": ch,
    }


_MARKETPLACE_MIGRATIONS: list[tuple[str, str]] = [
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
            self._migrate_drop_legacy_price_columns(conn)
            conn.commit()

    def _migrate_drop_legacy_price_columns(self, conn: sqlite3.Connection) -> None:
        """Drop retired ``median_price`` / ``highest_price`` / ``lowest_price``.

        One-time backfill: ``release_lowest_price`` inherits the COALESCE of
        legacy lowest/median when previously missing, so no listing floor
        is lost by the drop. Idempotent on subsequent opens.
        """
        existing = self._existing_columns(conn)
        legacy = [
            c for c in ("median_price", "highest_price", "lowest_price") if c in existing
        ]
        if not legacy:
            return
        if "lowest_price" in existing or "median_price" in existing:
            cols_for_backfill = ", ".join(
                c for c in ("lowest_price", "median_price") if c in existing
            )
            conn.execute(
                f"""
                UPDATE marketplace_stats
                   SET release_lowest_price = COALESCE(release_lowest_price, {cols_for_backfill})
                 WHERE release_lowest_price IS NULL
                """
            )
        for col in legacy:
            conn.execute(f"ALTER TABLE marketplace_stats DROP COLUMN {col}")

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

        When the stats payload omits ``num_for_sale``, ``release_payload`` fills
        it. Listing-floor data flows via ``release_lowest_price`` from
        ``extract_release_listing_fields`` (no legacy lowest/median columns).

        When ``price_suggestions_payload`` is ``{}`` but the row already has a
        non-empty ``price_suggestions_json``, the previous JSON is kept so the
        full per-condition ladder is not lost on transient API gaps.
        """
        rid = str(release_id).strip()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM marketplace_stats WHERE release_id = ?",
                (rid,),
            )
            prev_row = cur.fetchone()
            prev = self._row_to_dict(prev_row) if prev_row else {}

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

            conn.execute(
                """
                INSERT INTO marketplace_stats (
                    release_id, fetched_at, num_for_sale, blocked_from_sale,
                    raw_json, release_raw_json, price_suggestions_json,
                    release_lowest_price, release_num_for_sale,
                    community_want, community_have
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_id) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
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
                    comp["release_id"],
                    comp["fetched_at"],
                    comp["num_for_sale"],
                    comp["blocked_from_sale"],
                    comp["raw_json"],
                    comp["release_raw_json"],
                    comp["price_suggestions_json"],
                    comp["release_lowest_price"],
                    comp["release_num_for_sale"],
                    comp["community_want"],
                    comp["community_have"],
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
            "num_for_sale": row["num_for_sale"],
            "raw_json": row["raw_json"],
        }
        keys = row.keys()
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
