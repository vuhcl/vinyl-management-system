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
import copy
import json
import logging
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from grader.src.config_io import load_yaml_mapping
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


def _strip_seller_comment_prefixes(comments: str) -> str:
    """
    ``seller_comments`` from the sale export sometimes starts with
    ``Comments:``; strip once (case-insensitive) after trim.
    """
    s = (comments or "").strip()
    if not s:
        return s
    return re.sub(r"^comments:\s*", "", s, count=1, flags=re.IGNORECASE).strip()


def sale_row_to_inventory_listing(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """Map ``release_sale`` columns to keys ``parse_listing`` expects."""
    d = dict(row) if not isinstance(row, dict) else row
    rid = str(d.get("release_id") or "").strip()
    rh = str(d.get("row_hash") or "").strip()
    raw = (d.get("seller_comments") or "") or ""
    return {
        "id": f"{rid}:{rh}",
        "sleeve_condition": (d.get("sleeve_condition") or "") or "",
        "condition": (d.get("media_condition") or "") or "",
        "comments": _strip_seller_comment_prefixes(raw),
        "release": {},
    }


def iter_release_sale_rows(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    require_fetch_ok: bool = False,
    order_random: bool = False,
) -> Iterator[sqlite3.Row]:
    """
    Stream ``release_sale`` rows. When ``limit`` is set: ``order_random`` uses
    ``ORDER BY RANDOM()`` (expensive on large DBs); otherwise deterministic
    ``ORDER BY rs.release_id, rs.row_hash``. Clause order is always join → order → limit.
    """
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
        )
    else:
        sql = base

    use_limit = limit is not None and int(limit) > 0
    if use_limit and order_random:
        sql += "ORDER BY RANDOM() "
    else:
        sql += "ORDER BY rs.release_id, rs.row_hash "

    params: list[Any] = []
    if use_limit:
        sql += "LIMIT ?"
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


def _allocate_rows_across_media_strata(
    counts: dict[str, int],
    budget: int,
    rng: random.Random,
) -> dict[str, int]:
    """
    Choose integer take ``s[media]`` per stratum with ``sum(s) == budget``,
    ``0 <= s[k] <= counts[k]``, maximizing the minimum (as equal as caps allow).

    Used when capping rows per ``sleeve_label`` so dominant ``(sleeve, media)``
    pairs do not absorb the whole sleeve budget.
    """
    keys = [k for k, n in counts.items() if n > 0]
    if not keys or budget <= 0:
        return {k: 0 for k in counts}
    cap_total = sum(counts[k] for k in keys)
    r_rem = min(int(budget), cap_total)
    s_out: dict[str, int] = {k: 0 for k in counts}
    while r_rem > 0:
        active = [k for k in keys if s_out[k] < counts[k]]
        if not active:
            break
        per = r_rem // len(active)
        if per == 0:
            rng.shuffle(active)
            for k in active:
                if r_rem == 0:
                    break
                s_out[k] += 1
                r_rem -= 1
        else:
            distributed = 0
            for k in active:
                room = counts[k] - s_out[k]
                add = min(per, room)
                s_out[k] += add
                distributed += add
            r_rem -= distributed
    return s_out


def trim_sale_history_records_balanced(
    records: list[dict[str, Any]],
    *,
    max_rows_per_joint_grade: int | None,
    joint_sample_seed: int,
    max_rows_per_sleeve_grade: int | None = None,
    sleeve_stratum_sample_seed: int | None = None,
    max_total_sale_history_rows: int | None,
    balance_joint_within_sleeve_trim: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Cap sale-history export rows:

    1. Optional per ``(sleeve_label, media_label)`` stratum (``joint_sample_seed``).
    2. Optional per ``sleeve_label`` stratum (``sleeve_stratum_sample_seed``;
       default ``joint_sample_seed + 100003`` when omitted).
       When ``balance_joint_within_sleeve_trim`` is true, the sleeve cap splits
       its budget across distinct ``media_label`` values under that sleeve as
       evenly as row counts allow (then shuffles within each media bucket).
    3. Optional global cap by sorted ``item_id``.
    """
    stats: dict[str, Any] = {
        "input_rows": len(records),
        "after_joint_trim": len(records),
        "after_sleeve_trim": len(records),
        "after_total_trim": len(records),
        "strata_capped": 0,
        "sleeve_strata_capped": 0,
        "balance_joint_within_sleeve_trim": bool(balance_joint_within_sleeve_trim),
    }
    out = list(records)

    if max_rows_per_joint_grade is not None and int(max_rows_per_joint_grade) > 0:
        cap = int(max_rows_per_joint_grade)
        by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in out:
            sk = (
                str(r.get("sleeve_label") or ""),
                str(r.get("media_label") or ""),
            )
            by[sk].append(r)
        rng = random.Random(int(joint_sample_seed))
        trimmed: list[dict[str, Any]] = []
        strata_capped = 0
        for _k, group in by.items():
            g = list(group)
            rng.shuffle(g)
            if len(g) > cap:
                strata_capped += 1
            trimmed.extend(g[:cap])
        out = trimmed
        stats["after_joint_trim"] = len(out)
        stats["strata_capped"] = strata_capped
    else:
        stats["after_joint_trim"] = len(out)

    if max_rows_per_sleeve_grade is not None and int(max_rows_per_sleeve_grade) > 0:
        scap = int(max_rows_per_sleeve_grade)
        sleeve_seed = (
            int(sleeve_stratum_sample_seed)
            if sleeve_stratum_sample_seed is not None
            else int(joint_sample_seed) + 100_003
        )
        by_s: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in out:
            by_s[str(r.get("sleeve_label") or "")].append(r)
        rng_s = random.Random(sleeve_seed)
        trimmed_s: list[dict[str, Any]] = []
        sleeve_strata_capped = 0
        for _sl, group in by_s.items():
            g = list(group)
            if len(g) > scap:
                sleeve_strata_capped += 1
            if balance_joint_within_sleeve_trim and g:
                by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for r in g:
                    mk = str(r.get("media_label") or "")
                    by_m[mk].append(r)
                counts = {mk: len(rows) for mk, rows in by_m.items()}
                targets = _allocate_rows_across_media_strata(
                    counts, min(scap, len(g)), rng_s
                )
                for mk, take_n in targets.items():
                    if take_n <= 0:
                        continue
                    bucket = list(by_m.get(mk, []))
                    rng_s.shuffle(bucket)
                    trimmed_s.extend(bucket[:take_n])
            else:
                rng_s.shuffle(g)
                trimmed_s.extend(g[:scap])
        out = trimmed_s
        stats["after_sleeve_trim"] = len(out)
        stats["sleeve_strata_capped"] = sleeve_strata_capped
    else:
        stats["after_sleeve_trim"] = len(out)

    if max_total_sale_history_rows is not None and int(max_total_sale_history_rows) > 0:
        mxt = int(max_total_sale_history_rows)
        if len(out) > mxt:
            out = sorted(out, key=lambda r: str(r.get("item_id") or ""))[:mxt]
        stats["after_total_trim"] = len(out)
    else:
        stats["after_total_trim"] = len(out)

    return out, stats


def _sale_history_export_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Parse ``data.sale_history`` tuning keys (see grader.yaml)."""
    sh = (config.get("data") or {}).get("sale_history") or {}

    def _pos_int(key: str) -> int | None:
        v = sh.get(key)
        if v is None or v is False:
            return None
        try:
            i = int(v)
            return i if i > 0 else None
        except (TypeError, ValueError):
            return None

    order = str(sh.get("sql_sample_order") or "ordered").strip().lower()
    order_random = order == "random"
    seed_raw = sh.get("joint_sample_seed", 42)
    try:
        joint_seed = int(seed_raw)
    except (TypeError, ValueError):
        joint_seed = 42

    ss_raw = sh.get("sleeve_stratum_sample_seed")
    try:
        if ss_raw is None or ss_raw == "":
            sleeve_stratum_sample_seed = joint_seed + 100_003
        else:
            sleeve_stratum_sample_seed = int(ss_raw)
    except (TypeError, ValueError):
        sleeve_stratum_sample_seed = joint_seed + 100_003

    bal_raw = sh.get("balance_joint_within_sleeve_trim", False)
    if isinstance(bal_raw, str):
        balance_joint = bal_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        balance_joint = bool(bal_raw)

    return {
        "sql_prefetch_limit": _pos_int("sql_prefetch_limit"),
        "order_random": bool(order_random),
        "max_rows_per_joint_grade": _pos_int("max_rows_per_joint_grade"),
        "joint_sample_seed": joint_seed,
        "max_rows_per_sleeve_grade": _pos_int("max_rows_per_sleeve_grade"),
        "sleeve_stratum_sample_seed": sleeve_stratum_sample_seed,
        "max_total_sale_history_rows": _pos_int("max_total_sale_history_rows"),
        "balance_joint_within_sleeve_trim": balance_joint,
    }


def ingest_sale_history_records(
    sale_db: Path,
    ingester: DiscogsIngester,
    *,
    limit: int | None = None,
    require_fetch_ok: bool = False,
    order_random: bool = False,
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
            conn,
            limit=limit,
            require_fetch_ok=require_fetch_ok,
            order_random=order_random,
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
    Export sale SQLite → ``discogs_sale_history.jsonl`` with FS enrich, vinyl filter,
    and optional joint / global row caps (see ``data.sale_history`` in grader.yaml).

    ``limit`` overrides ``sql_prefetch_limit`` from config when ``limit > 0``.

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
    settings = _sale_history_export_settings(config)
    prefetch = (
        int(limit)
        if limit is not None and int(limit) > 0
        else settings["sql_prefetch_limit"]
    )
    order_random = settings["order_random"]

    ingester = DiscogsIngester(
        str(config_path),
        str(guidelines_path),
        offline_parse_only=True,
    )
    records, ingest_stats = ingest_sale_history_records(
        use_db,
        ingester,
        limit=prefetch,
        require_fetch_ok=require_fetch_ok,
        order_random=order_random,
    )
    records, post_stats = enrich_and_filter_sale_history_records(
        config, repo_root, records
    )
    records, trim_stats = trim_sale_history_records_balanced(
        records,
        max_rows_per_joint_grade=settings["max_rows_per_joint_grade"],
        joint_sample_seed=settings["joint_sample_seed"],
        max_rows_per_sleeve_grade=settings["max_rows_per_sleeve_grade"],
        sleeve_stratum_sample_seed=settings["sleeve_stratum_sample_seed"],
        max_total_sale_history_rows=settings["max_total_sale_history_rows"],
        balance_joint_within_sleeve_trim=settings[
            "balance_joint_within_sleeve_trim"
        ],
    )
    logger.info(
        "Sale history trim — rows_in=%s after_joint=%s after_sleeve=%s after_total=%s "
        "joint_strata_capped=%s sleeve_strata_capped=%s (joint_cap=%s sleeve_cap=%s "
        "total_cap=%s balance_sleeve_trim=%s)",
        trim_stats.get("input_rows"),
        trim_stats.get("after_joint_trim"),
        trim_stats.get("after_sleeve_trim"),
        trim_stats.get("after_total_trim"),
        trim_stats.get("strata_capped"),
        trim_stats.get("sleeve_strata_capped"),
        settings["max_rows_per_joint_grade"],
        settings["max_rows_per_sleeve_grade"],
        settings["max_total_sale_history_rows"],
        settings["balance_joint_within_sleeve_trim"],
    )
    out: dict[str, Any] = {
        "ok": True,
        "ingest": ingest_stats,
        "post": {k: int(v) for k, v in post_stats.items()},
        "trim": trim_stats,
        "sale_history_settings": {
            "sql_prefetch_limit": prefetch,
            "sql_sample_order": "random" if order_random else "ordered",
            "max_rows_per_joint_grade": settings["max_rows_per_joint_grade"],
            "joint_sample_seed": settings["joint_sample_seed"],
            "max_rows_per_sleeve_grade": settings["max_rows_per_sleeve_grade"],
            "sleeve_stratum_sample_seed": settings["sleeve_stratum_sample_seed"],
            "max_total_sale_history_rows": settings["max_total_sale_history_rows"],
            "balance_joint_within_sleeve_trim": settings[
                "balance_joint_within_sleeve_trim"
            ],
        },
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
        help="Override data.sale_history.sql_prefetch_limit: max SQLite rows (0 = use YAML)",
    )
    parser.add_argument(
        "--sql-sample-order",
        choices=("ordered", "random"),
        default=None,
        help="Override data.sale_history.sql_sample_order when --limit is set (default: YAML)",
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

    config = load_yaml_mapping(cfg_path)

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
    cfg_use = config
    if args.sql_sample_order is not None:
        cfg_use = copy.deepcopy(config)
        cfg_use.setdefault("data", {}).setdefault("sale_history", {})[
            "sql_sample_order"
        ] = args.sql_sample_order

    out = run_sale_history_ingest_from_config(
        cfg_use,
        cfg_path,
        guidelines_path,
        repo_root,
        limit=lim,
        require_fetch_ok=bool(args.require_fetch_ok),
        sale_db=sale_db,
        out_path=out_path,
        dry_run=args.dry_run,
    )
    if not out.get("ok"):
        print(out.get("error", out), file=sys.stderr)
        return 1
    logger.info(
        "Sale history ingest — rows_seen=%d saved=%d dropped=%d written=%d",
        out["ingest"]["total_rows"],
        out["ingest"]["saved"],
        out["ingest"]["dropped"],
        out.get("written", 0),
    )
    post = out.get("post") or {}
    logger.info(
        "Sale history after enrich+vinyl — fs_rows=%d api_releases=%d "
        "missing_dropped=%d vinyl_dropped=%d",
        post.get("enriched_from_feature_store", 0),
        post.get("enriched_from_discogs_api", 0),
        post.get("missing_format_dropped", 0),
        post.get("vinyl_dropped", 0),
    )
    if args.dry_run:
        logger.info("Dry run — not writing %s", out_path)
        return 0
    logger.info("Wrote %d records to %s", out.get("written", 0), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
