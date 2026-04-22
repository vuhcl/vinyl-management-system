#!/usr/bin/env python3
"""
Merge ``marketplace_stats.sqlite`` and ``sale_history.sqlite`` from multiple paths
(in-repo duplicates, teammate copies) into canonical outputs.

Policy matches VinylIQ plan §2a (MP): newer ``fetched_at`` as base, coalesce non-empty
from the other row, never wipe a non-empty ``price_suggestions_json`` with ``{}``.

Sale history: summary rows ``INSERT OR IGNORE`` on (release_id, fetched_at);
``release_sale`` upsert on (release_id, row_hash) with later ``fetched_at`` winning
then coalesce nulls; ``sale_history_fetch_status`` same style as MP.

Examples::

    PYTHONPATH=. uv run python price_estimator/scripts/merge_sqlite_sources.py \\
        --discover-repo-cache --dry-run

    PYTHONPATH=. uv run python price_estimator/scripts/merge_sqlite_sources.py \\
        --discover-repo-cache --apply

    PYTHONPATH=. uv run python price_estimator/scripts/merge_sqlite_sources.py \\
        --mp-merge PATH_FLAT PATH_NESTED PATH_TEAMMATE \\
        --sh-merge PATH_FLAT PATH_NESTED PATH_TEAMMATE \\
        --out-mp price_estimator/data/cache/marketplace_stats.sqlite \\
        --out-sh price_estimator/data/cache/sale_history.sqlite \\
        --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _repo_root() -> Path:
    # price_estimator/scripts/this.py -> parents[2] = monorepo root
    return Path(__file__).resolve().parents[2]


def _pe_package() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_fetched_at(raw: Any) -> datetime | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_blank_scalar(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _is_empty_ps_json(v: Any) -> bool:
    if _is_blank_scalar(v):
        return True
    s = str(v).strip()
    if s in ("{}", "null"):
        return True
    try:
        o = json.loads(s)
        return isinstance(o, dict) and len(o) == 0
    except json.JSONDecodeError:
        return False


def _mp_merge_symmetric(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two marketplace_stats row dicts (same release_id)."""
    ta = _parse_fetched_at(a.get("fetched_at"))
    tb = _parse_fetched_at(b.get("fetched_at"))
    if ta is None and tb is None:
        base, other = a, b
    elif ta is None:
        base, other = b, a
    elif tb is None:
        base, other = a, b
    elif ta > tb:
        base, other = a, b
    elif tb > ta:
        base, other = b, a
    else:
        # Equal timestamps: prefer higher non-null column count, then ``a``
        def nn(d: dict[str, Any]) -> int:
            return sum(1 for k, v in d.items() if k != "release_id" and v is not None)

        base, other = (a, b) if nn(a) >= nn(b) else (b, a)

    out = dict(base)
    other = dict(other)
    keys = set(out) | set(other)
    keys.discard("release_id")
    for k in keys:
        v = out.get(k)
        if k == "price_suggestions_json":
            if _is_empty_ps_json(v) and not _is_empty_ps_json(other.get(k)):
                out[k] = other[k]
            continue
        if _is_blank_scalar(v) and not _is_blank_scalar(other.get(k)):
            out[k] = other[k]
    # PS ladder: never replace non-empty with empty
    if _is_empty_ps_json(out.get("price_suggestions_json")) and not _is_empty_ps_json(
        other.get("price_suggestions_json")
    ):
        out["price_suggestions_json"] = other["price_suggestions_json"]

    # fetched_at = string from row with max parsed time
    if ta is None and tb is None:
        out["fetched_at"] = str(out.get("fetched_at") or other.get("fetched_at") or "")
    elif ta is None:
        out["fetched_at"] = str(b.get("fetched_at") or a.get("fetched_at") or "")
    elif tb is None:
        out["fetched_at"] = str(a.get("fetched_at") or b.get("fetched_at") or "")
    elif ta >= tb:
        out["fetched_at"] = str(a.get("fetched_at") or b.get("fetched_at") or "")
    else:
        out["fetched_at"] = str(b.get("fetched_at") or a.get("fetched_at") or "")

    rid = a.get("release_id") or b.get("release_id")
    if rid is not None:
        out["release_id"] = str(rid).strip()
    return out


def _mp_table_columns(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("PRAGMA table_info(marketplace_stats)")
    return [str(r[1]) for r in cur.fetchall()]


def _mp_upsert(conn: sqlite3.Connection, cols: list[str], row: dict[str, Any]) -> None:
    vals = [row.get(c) for c in cols]
    ph = ",".join("?" * len(cols))
    col_sql = ",".join(cols)
    conn.execute(
        f"INSERT OR REPLACE INTO marketplace_stats ({col_sql}) VALUES ({ph})",
        vals,
    )


def merge_marketplace_paths(paths: list[Path], out_path: Path, *, dry_run: bool) -> dict[str, int]:
    if len(paths) < 1:
        raise ValueError("need at least one marketplace_stats.sqlite path")
    print("MP merge inputs (absolute):")
    for p in paths:
        print(f"  {p.resolve()}")
    from price_estimator.src.storage.marketplace_db import MarketplaceStatsDB

    if dry_run:
        counts = []
        for p in paths:
            c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                n = int(c.execute("SELECT COUNT(*) FROM marketplace_stats").fetchone()[0])
            finally:
                c.close()
            counts.append(n)
        print(f"MP dry-run row counts per file: {counts}")
        return {"dry_run": 1, "paths": len(paths)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="mp_merge_", dir=str(out_path.parent)))
    tmp = tmp_dir / "merged.sqlite"
    shutil.copy2(paths[0], tmp)
    # Ensure schema on tmp
    MarketplaceStatsDB(tmp)

    conn = sqlite3.connect(str(tmp), timeout=120.0)
    conn.row_factory = sqlite3.Row
    try:
        cols = _mp_table_columns(conn)
        for src in paths[1:]:
            sconn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            sconn.row_factory = sqlite3.Row
            try:
                n_batch = 0
                for srow in sconn.execute("SELECT * FROM marketplace_stats"):
                    sdict = {k: srow[k] for k in srow.keys()}
                    rid = sdict.get("release_id")
                    if rid is None:
                        continue
                    cur2 = conn.execute(
                        "SELECT * FROM marketplace_stats WHERE release_id = ?", (rid,)
                    )
                    erow = cur2.fetchone()
                    if erow:
                        edict = {k: erow[k] for k in erow.keys()}
                        merged = _mp_merge_symmetric(edict, sdict)
                    else:
                        merged = dict(sdict)
                    _mp_upsert(conn, cols, merged)
                    n_batch += 1
                    if n_batch >= 5000:
                        conn.commit()
                        n_batch = 0
            finally:
                sconn.close()
        conn.commit()
        n_dup = int(
            conn.execute(
                "SELECT COUNT(*) FROM (SELECT release_id FROM marketplace_stats "
                "GROUP BY release_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        )
        if n_dup:
            raise RuntimeError(f"marketplace_stats duplicate release_id groups: {n_dup}")
    finally:
        conn.close()

    bak = out_path.with_suffix(out_path.suffix + ".pre_merge")
    if out_path.is_file():
        shutil.move(str(out_path), str(bak))
    shutil.move(str(tmp), str(out_path))
    if bak.is_file():
        bak.unlink(missing_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"MP merge wrote: {out_path.resolve()}")
    return {"paths": len(paths)}


def _sh_merge_fetch_status(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return _mp_merge_symmetric(a, b)


def _sh_merge_sale_row(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ta = _parse_fetched_at(a.get("fetched_at"))
    tb = _parse_fetched_at(b.get("fetched_at"))
    if ta is None and tb is None:
        base, other = a, b
    elif ta is None:
        base, other = b, a
    elif tb is None:
        base, other = a, b
    elif ta > tb:
        base, other = a, b
    elif tb > ta:
        base, other = b, a
    else:
        base, other = a, b
    out = dict(base)
    for k in set(out) | set(other):
        if k in ("release_id", "row_hash"):
            continue
        if _is_blank_scalar(out.get(k)) and not _is_blank_scalar(other.get(k)):
            out[k] = other[k]
    if ta is None and tb is None:
        out["fetched_at"] = str(out.get("fetched_at") or other.get("fetched_at") or "")
    elif ta is None:
        out["fetched_at"] = str(b.get("fetched_at") or a.get("fetched_at") or "")
    elif tb is None:
        out["fetched_at"] = str(a.get("fetched_at") or b.get("fetched_at") or "")
    elif ta >= tb:
        out["fetched_at"] = str(a.get("fetched_at") or b.get("fetched_at") or "")
    else:
        out["fetched_at"] = str(b.get("fetched_at") or a.get("fetched_at") or "")
    out["release_id"] = str(a.get("release_id") or b.get("release_id") or "").strip()
    out["row_hash"] = str(a.get("row_hash") or b.get("row_hash") or "").strip()
    return out


def merge_sale_history_paths(paths: list[Path], out_path: Path, *, dry_run: bool) -> dict[str, int]:
    if len(paths) < 1:
        raise ValueError("need at least one sale_history.sqlite path")
    print("SH merge inputs (absolute):")
    for p in paths:
        print(f"  {p.resolve()}")
    from price_estimator.src.storage.sale_history_db import SaleHistoryDB

    if dry_run:
        for p in paths:
            c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                n1 = int(c.execute("SELECT COUNT(*) FROM release_sale").fetchone()[0])
                n2 = int(c.execute("SELECT COUNT(*) FROM sale_history_fetch_status").fetchone()[0])
            finally:
                c.close()
            print(f"  {p}: release_sale={n1:,} fetch_status={n2:,}")
        return {"dry_run": 1}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="sh_merge_", dir=str(out_path.parent)))
    tmp = tmp_dir / "merged.sqlite"
    shutil.copy2(paths[0], tmp)
    SaleHistoryDB(tmp)

    conn = sqlite3.connect(str(tmp), timeout=120.0)
    conn.row_factory = sqlite3.Row
    try:
        for src in paths[1:]:
            sconn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            sconn.row_factory = sqlite3.Row
            try:
                # release_sale_summary
                for row in sconn.execute("SELECT * FROM release_sale_summary"):
                    d = {k: row[k] for k in row.keys()}
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO release_sale_summary (
                            release_id, fetched_at, last_sold_on, average, median, high, low, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            d["release_id"],
                            d["fetched_at"],
                            d.get("last_sold_on"),
                            d.get("average"),
                            d.get("median"),
                            d.get("high"),
                            d.get("low"),
                            d["raw_json"],
                        ),
                    )
                # release_sale (dynamic columns for optional price_user_usd_approx)
                rcols = [r[1] for r in conn.execute("PRAGMA table_info(release_sale)").fetchall()]
                for row in sconn.execute("SELECT * FROM release_sale"):
                    d = {k: row[k] for k in row.keys()}
                    rid, rh = d["release_id"], d["row_hash"]
                    cur = conn.execute(
                        "SELECT * FROM release_sale WHERE release_id = ? AND row_hash = ?",
                        (rid, rh),
                    )
                    ex = cur.fetchone()
                    if ex:
                        exd = {k: ex[k] for k in ex.keys()}
                        merged = _sh_merge_sale_row(exd, d)
                    else:
                        merged = dict(d)
                    merged = {k: merged.get(k) for k in rcols}
                    vals = [merged.get(c) for c in rcols]
                    ph = ",".join("?" * len(rcols))
                    conn.execute(
                        f"INSERT OR REPLACE INTO release_sale ({','.join(rcols)}) VALUES ({ph})",
                        vals,
                    )
                # sale_history_fetch_status
                fcols = [r[1] for r in conn.execute("PRAGMA table_info(sale_history_fetch_status)").fetchall()]
                for row in sconn.execute("SELECT * FROM sale_history_fetch_status"):
                    d = {k: row[k] for k in row.keys()}
                    rid = d["release_id"]
                    cur = conn.execute(
                        "SELECT * FROM sale_history_fetch_status WHERE release_id = ?", (rid,)
                    )
                    ex = cur.fetchone()
                    if ex:
                        exd = {k: ex[k] for k in ex.keys()}
                        merged = _sh_merge_fetch_status(exd, d)
                    else:
                        merged = dict(d)
                    merged = {k: merged.get(k) for k in fcols}
                    vals = [merged.get(c) for c in fcols]
                    ph = ",".join("?" * len(fcols))
                    conn.execute(
                        f"INSERT OR REPLACE INTO sale_history_fetch_status ({','.join(fcols)}) VALUES ({ph})",
                        vals,
                    )
                conn.commit()
            finally:
                sconn.close()
        conn.commit()
    finally:
        conn.close()

    bak = out_path.with_suffix(out_path.suffix + ".pre_merge")
    if out_path.is_file():
        shutil.move(str(out_path), str(bak))
    shutil.move(str(tmp), str(out_path))
    if bak.is_file():
        bak.unlink(missing_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"SH merge wrote: {out_path.resolve()}")
    return {"paths": len(paths)}


def _discovered_mp_paths() -> list[Path]:
    """Nested (legacy duplicate tree) first when present, then flat — fold order for in-repo merge."""
    root = _repo_root()
    nested = root / "price_estimator" / "price_estimator" / "data" / "cache" / "marketplace_stats.sqlite"
    flat = root / "price_estimator" / "data" / "cache" / "marketplace_stats.sqlite"
    out: list[Path] = []
    if nested.is_file():
        out.append(nested)
    if flat.is_file():
        out.append(flat)
    return out


def _discovered_sh_paths() -> list[Path]:
    """Flat (canonical large DB) first, then nested legacy path when present."""
    root = _repo_root()
    flat = root / "price_estimator" / "data" / "cache" / "sale_history.sqlite"
    nested = root / "price_estimator" / "price_estimator" / "data" / "cache" / "sale_history.sqlite"
    out: list[Path] = []
    if flat.is_file():
        out.append(flat)
    if nested.is_file():
        out.append(nested)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mp-merge", type=Path, nargs="*", default=None, help="Ordered marketplace DB paths")
    ap.add_argument("--sh-merge", type=Path, nargs="*", default=None, help="Ordered sale_history DB paths")
    ap.add_argument(
        "--out-mp",
        type=Path,
        default=None,
        help="Output marketplace_stats.sqlite (default: flat package cache)",
    )
    ap.add_argument(
        "--out-sh",
        type=Path,
        default=None,
        help="Output sale_history.sqlite (default: flat package cache)",
    )
    ap.add_argument("--discover-repo-cache", action="store_true", help="Use flat then nested in-repo caches")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.dry_run == args.apply:
        print("Specify exactly one of --dry-run or --apply", file=sys.stderr)
        return 2

    pkg = _pe_package()
    default_out_mp = pkg / "data" / "cache" / "marketplace_stats.sqlite"
    default_out_sh = pkg / "data" / "cache" / "sale_history.sqlite"
    out_mp = args.out_mp or default_out_mp
    out_sh = args.out_sh or default_out_sh

    mp_paths = None if args.mp_merge is None else list(args.mp_merge)
    sh_paths = None if args.sh_merge is None else list(args.sh_merge)
    if args.discover_repo_cache:
        if not mp_paths:
            mp_paths = _discovered_mp_paths()
        if not sh_paths:
            sh_paths = _discovered_sh_paths()

    if not mp_paths or not sh_paths:
        print("Need --mp-merge and --sh-merge paths or --discover-repo-cache", file=sys.stderr)
        return 2

    dry = bool(args.dry_run)
    merge_marketplace_paths(mp_paths, out_mp, dry_run=dry)
    merge_sale_history_paths(sh_paths, out_sh, dry_run=dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
