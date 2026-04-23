"""
grader/src/data/ingest_sale_history.py

Read completed-sale rows from ``price_estimator`` sale-history SQLite
(``release_sale``) and write ``discogs_sale_history.jsonl`` in the same record
shape as ``ingest_discogs.parse_listing`` (for ``harmonize_labels``).

Uses ``DiscogsIngester(..., offline_parse_only=True)`` so grade maps, boilerplate
stripping, and drop rules match inventory ingestion without ``DISCOGS_TOKEN``.

Usage (repo root):
    python -m grader.src.data.ingest_sale_history
    python -m grader.src.data.ingest_sale_history --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator

from grader.src.data.ingest_discogs import DiscogsIngester
from grader.src.data.vinyl_format import (
    DISCOGS_SALE_HISTORY_SOURCE,
    filter_records_vinyl_by_source,
    format_fields_from_releases_features,
)

logger = logging.getLogger(__name__)

# Backwards compatibility — same as ``vinyl_format.DISCOGS_SALE_HISTORY_SOURCE``.
SALE_HISTORY_SOURCE = DISCOGS_SALE_HISTORY_SOURCE


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def sale_row_to_inventory_listing(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """Map ``release_sale`` columns to keys ``parse_listing`` expects."""
    d = dict(row) if not isinstance(row, dict) else row
    rid = str(d.get("release_id") or "").strip()
    rh = str(d.get("row_hash") or "").strip()
    return {
        "id": f"{rid}:{rh}",
        "sleeve_condition": (d.get("sleeve_condition") or "") or "",
        "condition": (d.get("media_condition") or "") or "",
        "comments": (d.get("seller_comments") or "") or "",
        "release": {},
    }


def iter_release_sale_rows(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    require_fetch_ok: bool = False,
) -> Iterator[sqlite3.Row]:
    base = (
        "SELECT rs.release_id, rs.row_hash, rs.order_date, rs.media_condition, "
        "rs.sleeve_condition, rs.seller_comments "
        "FROM release_sale AS rs "
    )
    if require_fetch_ok:
        sql = (
            base
            + "INNER JOIN sale_history_fetch_status AS st "
            "ON st.release_id = rs.release_id AND st.status = 'ok' "
            "ORDER BY rs.release_id, rs.row_hash"
        )
    else:
        sql = base + "ORDER BY rs.release_id, rs.row_hash"
    params: list[Any] = []
    if limit is not None and int(limit) > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    yield from conn.execute(sql, params)


def _release_id_from_item_id(item_id: str) -> str:
    s = (item_id or "").strip()
    if ":" in s:
        return s.split(":", 1)[0].strip()
    return s


def _load_releases_format_map(
    feature_store: Path, release_ids: set[str]
) -> dict[str, tuple[str | None, str | None]]:
    if not feature_store.is_file() or not release_ids:
        return {}
    out: dict[str, tuple[str | None, str | None]] = {}
    conn = sqlite3.connect(str(feature_store))
    try:
        ids = list(release_ids)
        chunk = 500
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            qmarks = ",".join("?" * len(part))
            cur = conn.execute(
                f"SELECT release_id, format_desc, formats_json FROM releases_features "
                f"WHERE release_id IN ({qmarks})",
                part,
            )
            for r in cur:
                out[str(r[0])] = (r[1], r[2])
    finally:
        conn.close()
    return out


def _enrich_from_discogs_api(
    release_ids: set[str],
) -> dict[str, tuple[str | None, str | None]]:
    if not release_ids:
        return {}
    try:
        from shared.discogs_api.client import discogs_client_from_env
    except ImportError:  # pragma: no cover
        logger.warning("Discogs client unavailable for sale_history API enrich.")
        return {}
    client = discogs_client_from_env()
    if client is None:
        logger.warning(
            "enrich_missing_from_discogs: no token — set DISCOGS_TOKEN for fallback."
        )
        return {}
    out: dict[str, tuple[str | None, str | None]] = {}
    for rid in sorted(release_ids):
        try:
            data = client.get_release(rid)
        except Exception as exc:  # noqa: BLE001
            logger.info("get_release failed for %s: %s", rid, exc)
            continue
        if not isinstance(data, dict):
            continue
        fm = data.get("formats")
        raw: str
        if isinstance(fm, list):
            raw = json.dumps(fm, ensure_ascii=False)
        else:
            raw = ""
        if not str(raw).strip():
            continue
        # Store like ``formats_json`` so the main pass calls ``format_fields_…`` once.
        out[rid] = (None, raw)
    return out


def enrich_and_filter_sale_history_records(
    config: dict[str, Any],
    repo_root: Path,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Join ``release_id`` to ``releases_features`` (and optional API fallback);
    set ``release_format`` / ``release_description``; run vinyl allowlist filter.

    Uses ``data.sale_history`` (see grader.yaml).
    """
    sh = (config.get("data") or {}).get("sale_history") or {}
    stats: dict[str, int] = {
        "enriched_from_feature_store": 0,
        "enriched_from_discogs_api": 0,
        "missing_format_dropped": 0,
        "vinyl_dropped": 0,
    }

    enrich_fs = bool(sh.get("enrich_from_feature_store", True))
    enrich_api = bool(sh.get("enrich_missing_from_discogs", False))
    on_missing = str(sh.get("on_missing_release", "keep") or "keep").strip().lower()
    if on_missing not in ("keep", "drop"):
        on_missing = "keep"
    do_filter = bool(sh.get("apply_vinyl_filter", True))

    raw_fs = (sh.get("feature_store_path") or "").strip()
    feature_store = Path(raw_fs) if raw_fs else Path()
    if raw_fs and not feature_store.is_absolute():
        feature_store = (repo_root / feature_store).resolve()
    use_fs = bool(enrich_fs and raw_fs and feature_store.is_file())

    release_ids: set[str] = {
        _release_id_from_item_id(str(r.get("item_id") or "")) for r in records
    }
    release_ids.discard("")

    fmt_by_rid: dict[str, tuple[str | None, str | None]] = {}
    from_feature_store: set[str] = set()
    if use_fs:
        fmt_by_rid = _load_releases_format_map(feature_store, release_ids)
        from_feature_store = set(fmt_by_rid.keys())

    if enrich_api and release_ids:
        need_api: set[str] = set()
        for rid in release_ids:
            fd, fj = fmt_by_rid.get(rid, (None, None))
            has_raw = (str(fd or "").strip()) or (str(fj or "").strip())
            if not has_raw:
                need_api.add(rid)
        if need_api:
            extra = _enrich_from_discogs_api(need_api)
            stats["enriched_from_discogs_api"] = len(extra)
            for k, v in extra.items():
                fmt_by_rid[k] = v

    kept: list[dict[str, Any]] = []
    for rec in records:
        rec = dict(rec)
        rid = _release_id_from_item_id(str(rec.get("item_id") or ""))
        fdesc, fj = fmt_by_rid.get(rid, (None, None))
        if use_fs and rid and rid in from_feature_store:
            stats["enriched_from_feature_store"] += 1
        rf, rd = format_fields_from_releases_features(
            str(fdesc or ""), str(fj or "")
        )
        rec["release_format"] = rf
        rec["release_description"] = rd

        empty_fmt = not rf.strip() and not rd.strip()
        if on_missing == "drop" and empty_fmt:
            stats["missing_format_dropped"] += 1
            continue
        kept.append(rec)

    if not do_filter:
        return kept, stats

    filtered, vinyl_d = filter_records_vinyl_by_source(
        kept, source_allowlist={DISCOGS_SALE_HISTORY_SOURCE}
    )
    stats["vinyl_dropped"] = vinyl_d
    return filtered, stats


def ingest_sale_history_records(
    sale_db: Path,
    ingester: DiscogsIngester,
    *,
    limit: int | None = None,
    require_fetch_ok: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Load sale rows, map through ``parse_listing``, set ``source`` to
    ``discogs_sale_history``. Returns (records, stats) with keys
    total_rows, saved, dropped.
    """
    stats = {"total_rows": 0, "saved": 0, "dropped": 0}
    out: list[dict[str, Any]] = []
    # ``parse_listing`` increments ``_stats["drops"]``; ``run()`` usually
    # initializes this — set up for offline-only callers.
    if "drops" not in ingester._stats:
        ingester._stats["drops"] = {}

    conn = sqlite3.connect(str(sale_db))
    conn.row_factory = sqlite3.Row
    try:
        for row in iter_release_sale_rows(
            conn, limit=limit, require_fetch_ok=require_fetch_ok
        ):
            stats["total_rows"] += 1
            listing = sale_row_to_inventory_listing(row)
            rec = ingester.parse_listing(listing)
            if rec is None:
                stats["dropped"] += 1
                continue
            rec["source"] = SALE_HISTORY_SOURCE
            out.append(rec)
            stats["saved"] += 1
    finally:
        conn.close()
    return out, stats


def _default_paths_from_config(config: dict[str, Any], repo_root: Path) -> tuple[Path, Path]:
    paths_cfg = config.get("paths") or {}
    processed = paths_cfg.get("processed", "grader/data/processed")
    processed_dir = Path(processed)
    if not processed_dir.is_absolute():
        processed_dir = (repo_root / processed_dir).resolve()

    sh = (config.get("data") or {}).get("sale_history") or {}
    raw_sqlite = sh.get(
        "sqlite_path",
        "price_estimator/data/cache/sale_history.sqlite",
    )
    sale_db = Path(raw_sqlite)
    if not sale_db.is_absolute():
        sale_db = (repo_root / sale_db).resolve()

    raw_jsonl = sh.get("processed_jsonl")
    if raw_jsonl:
        out_path = Path(raw_jsonl)
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()
    else:
        out_path = (processed_dir / "discogs_sale_history.jsonl").resolve()
    return sale_db, out_path


def run_sale_history_ingest_from_config(
    config: dict[str, Any],
    config_path: Path,
    guidelines_path: Path,
    repo_root: Path,
    *,
    limit: int | None = None,
    require_fetch_ok: bool = False,
    sale_db: Path | None = None,
    out_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Export sale SQLite → ``discogs_sale_history.jsonl`` with FS enrich and vinyl filter.

    Used by ``grader.src.pipeline`` (optional step) and mirrors ``main()`` defaults.
    """
    default_db, default_out = _default_paths_from_config(config, repo_root)
    use_db = sale_db or default_db
    use_out = out_path or default_out
    if not use_db.is_absolute():
        use_db = (repo_root / use_db).resolve()
    if not use_out.is_absolute():
        use_out = (repo_root / use_out).resolve()
    if not use_db.is_file():
        return {"ok": False, "error": "sale_db_missing", "path": str(use_db)}
    ingester = DiscogsIngester(
        str(config_path),
        str(guidelines_path),
        offline_parse_only=True,
    )
    records, ingest_stats = ingest_sale_history_records(
        use_db, ingester, limit=limit, require_fetch_ok=require_fetch_ok
    )
    records, post_stats = enrich_and_filter_sale_history_records(
        config, repo_root, records
    )
    out: dict[str, Any] = {
        "ok": True,
        "ingest": ingest_stats,
        "post": {k: int(v) for k, v in post_stats.items()},
        "out": str(use_out),
        "written": len(records) if not dry_run else 0,
        "dry_run": dry_run,
    }
    if dry_run:
        return out
    use_out.parent.mkdir(parents=True, exist_ok=True)
    with open(use_out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Export sale_history SQLite rows to grader JSONL (discogs_sale_history)",
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    parser.add_argument(
        "--sale-db",
        type=Path,
        default=None,
        help="Sale history SQLite (default: data.sale_history.sqlite_path in config)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (default: data.sale_history.processed_jsonl in config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows from release_sale (0 = all)",
    )
    parser.add_argument(
        "--require-fetch-ok",
        action="store_true",
        help="Only rows whose release_id has sale_history_fetch_status.status='ok'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count but do not write output file",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_here()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    guidelines_path = Path(args.guidelines)
    if not guidelines_path.is_absolute():
        guidelines_path = (repo_root / guidelines_path).resolve()

    import yaml

    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    default_sale_db, default_out = _default_paths_from_config(config, repo_root)
    sale_db = args.sale_db or default_sale_db
    out_path = args.out or default_out
    if not sale_db.is_absolute():
        sale_db = (repo_root / sale_db).resolve()
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    if not sale_db.is_file():
        print(f"Sale history database not found: {sale_db}", file=sys.stderr)
        return 1

    lim = int(args.limit) if args.limit and args.limit > 0 else None
    ingester = DiscogsIngester(
        str(cfg_path),
        str(guidelines_path),
        offline_parse_only=True,
    )
    records, stats = ingest_sale_history_records(
        sale_db,
        ingester,
        limit=lim,
        require_fetch_ok=bool(args.require_fetch_ok),
    )
    logger.info(
        "Sale history ingest — rows_seen=%d saved=%d dropped=%d",
        stats["total_rows"],
        stats["saved"],
        stats["dropped"],
    )
    records, post = enrich_and_filter_sale_history_records(config, repo_root, records)
    logger.info(
        "Sale history after enrich+vinyl — fs_rows=%d api_releases=%d "
        "missing_dropped=%d vinyl_dropped=%d out_rows=%d",
        post.get("enriched_from_feature_store", 0),
        post.get("enriched_from_discogs_api", 0),
        post.get("missing_format_dropped", 0),
        post.get("vinyl_dropped", 0),
        len(records),
    )
    if args.dry_run:
        logger.info("Dry run — not writing %s", out_path)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
