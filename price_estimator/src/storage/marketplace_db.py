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


def extract_marketplace_stats_listing(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Listing scalars from ``GET /marketplace/stats/{release_id}``.

    Discogs documents ``lowest_price`` (often ``{value, currency}``),
    ``num_for_sale``, and ``blocked_from_sale``. Community totals are not in
    this response; pass ``community_want`` / ``community_have`` on ``payload``
    when upserting merged data (e.g. from ``GET /releases/{id}/stats``).
    """
    if not isinstance(payload, dict):
        return {}
    lp_raw = payload.get("lowest_price")
    lp = _parse_price_field(lp_raw if isinstance(lp_raw, dict) else lp_raw)
    nfs = payload.get("num_for_sale")
    if nfs is None:
        nfs = payload.get("for_sale_count")
    try:
        rns = int(nfs) if nfs is not None else None
    except (TypeError, ValueError):
        rns = None
    return {
        "release_lowest_price": lp,
        "release_num_for_sale": rns,
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


def price_suggestion_usd_for_grade_label(
    ladder: dict[str, float],
    label: str | None,
) -> float | None:
    """
    Map a user / Discogs grade string onto a ladder USD value.

    Tries exact key match first, then case-insensitive containment between label and
    canonical ladder keys (``"Near Mint (NM or M-)"``, ``"Good (G)"``, ...).
    """
    if not ladder or label is None or not str(label).strip():
        return None
    t_raw = str(label).strip()
    if t_raw in ladder:
        return float(ladder[t_raw])

    tl = t_raw.lower()
    for lk, fv in ladder.items():
        lk_s = str(lk).strip()
        if lk_s.lower() == tl:
            return float(fv)
    # Substring heuristic (handles minor punctuation / spacing drift).
    for lk, fv in ladder.items():
        lk_l = str(lk).strip().lower()
        if lk_l in tl or tl in lk_l:
            return float(fv)
    return None


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

    rlj = rel_raw if rel_raw is not None else prev.get("release_raw_json")
    psj = ps_raw if ps_raw is not None else prev.get("price_suggestions_json")

    rlp = rel_ext.get("release_lowest_price")
    rns = rel_ext.get("release_num_for_sale")
    cw = rel_ext.get("community_want")
    ch = rel_ext.get("community_have")
    if release_payload is None:
        mstat = extract_marketplace_stats_listing(payload)
        if mstat.get("release_lowest_price") is not None:
            rlp = mstat["release_lowest_price"]
        else:
            rlp = prev.get("release_lowest_price")
        # ``release_num_for_sale`` mirrors GET /releases; do not derive it from a
        # bare marketplace ``num_for_sale`` fragment alone (infer depth uses
        # ``norm[num_for_sale]`` separately).
        nfs_m = mstat.get("release_num_for_sale")
        if (
            mstat.get("release_lowest_price") is not None
            and nfs_m is not None
        ):
            rns = nfs_m
        else:
            rns = prev.get("release_num_for_sale")

        cw_in = payload.get("community_want") if isinstance(payload, dict) else None
        ch_in = payload.get("community_have") if isinstance(payload, dict) else None
        cw = None
        ch = None
        if cw_in is not None:
            try:
                cw = int(cw_in)
            except (TypeError, ValueError):
                cw = None
        if ch_in is not None:
            try:
                ch = int(ch_in)
            except (TypeError, ValueError):
                ch = None
        if cw is None:
            cw = prev.get("community_want")
        if ch is None:
            ch = prev.get("community_have")
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


# --- Inference / Redis: aligned projections (train ↔ serve parity) ---

MARKETPLACE_INFERENCE_STATS_KEYS = (
    "release_lowest_price",
    "num_for_sale",
    "release_num_for_sale",
    "community_want",
    "community_have",
    "blocked_from_sale",
    "price_suggestions_json",
    "sale_stats_average_usd",
    "sale_stats_median_usd",
    "sale_stats_high_usd",
    "sale_stats_low_usd",
)

REDIS_MP_STATS_SCHEMA_VER = 2


def marketplace_inference_stats_from_row(row: dict[str, Any] | None) -> dict[str, Any]:
    """Normalized marketplace slice feeding ``row_dict_for_inference`` + anchors."""
    src = dict(row or {})
    rlp = _parse_price_field(src.get("release_lowest_price"))
    nfs = src.get("num_for_sale")
    try:
        num_sale = int(nfs) if nfs is not None else 0
    except (TypeError, ValueError):
        num_sale = 0
    rnfs = src.get("release_num_for_sale")
    try:
        release_nfs = int(rnfs) if rnfs is not None else None
    except (TypeError, ValueError):
        release_nfs = None
    cw_raw = src.get("community_want")
    try:
        cw = int(cw_raw) if cw_raw is not None else None
    except (TypeError, ValueError):
        cw = None
    ch_raw = src.get("community_have")
    try:
        ch = int(ch_raw) if ch_raw is not None else None
    except (TypeError, ValueError):
        ch = None
    bl = src.get("blocked_from_sale")
    blocked: int | None
    try:
        if bl is None:
            blocked = None
        else:
            blocked = 1 if int(bl) else 0
    except (TypeError, ValueError):
        blocked = None
    psj = src.get("price_suggestions_json")
    if isinstance(psj, dict):
        ps_out = json.dumps(psj, ensure_ascii=False, separators=(",", ":"))
    elif psj is None or not str(psj).strip():
        ps_out = None
    else:
        ps_out = str(psj).strip()

    s_avg = _parse_price_field(src.get("sale_stats_average_usd"))
    s_med = _parse_price_field(src.get("sale_stats_median_usd"))
    s_hi = _parse_price_field(src.get("sale_stats_high_usd"))
    s_lo = _parse_price_field(src.get("sale_stats_low_usd"))

    out: dict[str, Any] = {
        "release_lowest_price": rlp,
        "num_for_sale": num_sale,
        "release_num_for_sale": release_nfs,
        "community_want": cw if cw is not None else None,
        "community_have": ch if ch is not None else None,
        "blocked_from_sale": blocked,
        "price_suggestions_json": ps_out,
        "sale_stats_average_usd": s_avg,
        "sale_stats_median_usd": s_med,
        "sale_stats_high_usd": s_hi,
        "sale_stats_low_usd": s_lo,
    }
    return out


def redis_marketplace_cache_blob_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable Redis value (includes schema version marker)."""
    core = marketplace_inference_stats_from_row(row)
    blob: dict[str, Any] = {**core}
    blob["_schema_ver"] = REDIS_MP_STATS_SCHEMA_VER
    return blob


def decode_redis_marketplace_cached_payload(cached: dict[str, Any]) -> dict[str, Any]:
    """Hydrate inference stats dict from Redis JSON (handles legacy skinny payloads)."""
    if cached.get("_schema_ver") == REDIS_MP_STATS_SCHEMA_VER:
        raw_pre = {k: cached.get(k) for k in MARKETPLACE_INFERENCE_STATS_KEYS}
        return marketplace_inference_stats_from_row(raw_pre)

    raw: dict[str, Any] = {k: None for k in MARKETPLACE_INFERENCE_STATS_KEYS}
    raw["release_lowest_price"] = cached.get("release_lowest_price")
    nfs = cached.get("num_for_sale")
    raw["num_for_sale"] = int(nfs) if nfs is not None else 0
    return marketplace_inference_stats_from_row(raw)


def merge_marketplace_client_overlay(
    base: dict[str, Any],
    overlay: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge scraped / client-visible Discogs marketplace fields onto ``base``."""

    def _blocked_int(v: Any) -> int:
        return 1 if bool(int(v)) else 0

    if not overlay:
        return base
    out = dict(base)
    for k in (
        "release_lowest_price",
        "num_for_sale",
        "release_num_for_sale",
        "community_want",
        "community_have",
    ):
        if k not in overlay or overlay[k] is None:
            continue
        if k == "release_lowest_price":
            fv = _parse_price_field(overlay[k])
            if fv is None:
                continue
            out[k] = float(fv)
            continue
        try:
            out[k] = int(overlay[k])
        except (TypeError, ValueError):
            continue

    if "blocked_from_sale" in overlay and overlay["blocked_from_sale"] is not None:
        v = overlay["blocked_from_sale"]
        if isinstance(v, bool):
            out["blocked_from_sale"] = 1 if v else 0
        else:
            try:
                out["blocked_from_sale"] = _blocked_int(v)
            except (TypeError, ValueError):
                pass

    if "price_suggestions_json" in overlay and overlay["price_suggestions_json"] is not None:
        ps = overlay["price_suggestions_json"]
        if isinstance(ps, dict):
            if ps:
                out["price_suggestions_json"] = json.dumps(
                    ps,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        elif isinstance(ps, str) and str(ps).strip():
            out["price_suggestions_json"] = str(ps).strip()

    for sale_k in (
        "sale_stats_average_usd",
        "sale_stats_median_usd",
        "sale_stats_high_usd",
        "sale_stats_low_usd",
    ):
        if sale_k not in overlay or overlay[sale_k] is None:
            continue
        fv = _parse_price_field(overlay[sale_k])
        if fv is None:
            continue
        out[sale_k] = float(fv)

    return marketplace_inference_stats_from_row(out)


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
