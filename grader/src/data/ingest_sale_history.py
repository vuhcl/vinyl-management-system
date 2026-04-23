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

logger = logging.getLogger(__name__)

SALE_HISTORY_SOURCE = "discogs_sale_history"


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
