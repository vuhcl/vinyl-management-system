#!/usr/bin/env python3
"""Audit overlap counts across feature_store, marketplace_stats, and sale_history.

With ``--format-audit``, also reports missing ``formats_json`` / ``format_desc`` on
FS rows joined to marketplace (global and an approximate cheap slice: MP anchor ≤ $15).

With ``--check``, exit 1 if thresholds fail (see ``--min-mp-in-fs-ratio``,
``--min-fs-mp-sh-ratio``, ``--max-format-both-empty-pct``). Run after collectors
and before training.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, p: Path | None, fallback: str) -> Path:
    x = p or Path(fallback)
    if not x.is_absolute():
        x = root / x
    return x


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0] or 0)


def _format_field_audit(conn: sqlite3.Connection) -> tuple[int, float]:
    """FS∩MP: missing ``formats_json`` / ``format_desc`` (hurts format_medium_flags)."""
    print("--- format field audit (FS inner join MP) ---")
    row = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN f.formats_json IS NULL OR TRIM(f.formats_json) IN ('', '[]')
                   THEN 1 ELSE 0 END) AS empty_fj,
               SUM(CASE WHEN f.format_desc IS NULL OR TRIM(f.format_desc) = ''
                   THEN 1 ELSE 0 END) AS empty_fd,
               SUM(CASE WHEN (
                   (f.formats_json IS NULL OR TRIM(f.formats_json) IN ('', '[]'))
                   AND (f.format_desc IS NULL OR TRIM(f.format_desc) = '')
               ) THEN 1 ELSE 0 END) AS empty_both
        FROM fs.releases_features f
        INNER JOIN mp.marketplace_stats m ON m.release_id = f.release_id
        """
    ).fetchone()
    n = int(row[0] or 0)
    if n <= 0:
        print("  (no FS∩MP rows)")
        return 0, 0.0
    efj, efd, eb = int(row[1] or 0), int(row[2] or 0), int(row[3] or 0)
    both_pct = 100.0 * eb / n
    print(f"  FS∩MP rows: {n:,}")
    print(f"  empty formats_json: {efj:,} ({100.0 * efj / n:.2f}%)")
    print(f"  empty format_desc:  {efd:,} ({100.0 * efd / n:.2f}%)")
    print(f"  both empty:         {eb:,} ({both_pct:.2f}%)")
    cheap = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN f.formats_json IS NULL OR TRIM(f.formats_json) IN ('', '[]')
                   THEN 1 ELSE 0 END),
               SUM(CASE WHEN f.format_desc IS NULL OR TRIM(f.format_desc) = ''
                   THEN 1 ELSE 0 END)
        FROM fs.releases_features f
        INNER JOIN mp.marketplace_stats m ON m.release_id = f.release_id
        WHERE m.release_lowest_price IS NOT NULL
          AND m.release_lowest_price > 0
          AND m.release_lowest_price <= 15.0
        """
    ).fetchone()
    nc = int(cheap[0] or 0)
    if nc > 0:
        cj, cd = int(cheap[1] or 0), int(cheap[2] or 0)
        print(
            f"  cheap slice (MP anchor ≤ $15): n={nc:,} | "
            f"empty formats_json {100.0 * cj / nc:.1f}% | "
            f"empty format_desc {100.0 * cd / nc:.1f}%"
        )
    return n, both_pct


def _check_thresholds(
    *,
    n_fs: int,
    n_mp: int,
    fs_mp: int,
    has_sh: bool,
    fs_mp_sh: int,
    format_both_empty_pct: float,
    min_mp_in_fs_ratio: float,
    min_fs_mp_sh_ratio: float,
    max_format_both_empty_pct: float,
) -> list[str]:
    """Return human-readable failure messages (empty if all checks pass)."""
    errors: list[str] = []
    if n_fs <= 0:
        errors.append("feature_store has no releases_features rows")
    if n_mp <= 0:
        errors.append("marketplace_stats is empty")
    if n_mp > 0:
        mp_in_fs = fs_mp / n_mp
        if mp_in_fs < min_mp_in_fs_ratio:
            errors.append(
                f"FS∩MP/MP={mp_in_fs:.4f} < --min-mp-in-fs-ratio {min_mp_in_fs_ratio}"
            )
    if has_sh and fs_mp > 0:
        sh_cov = fs_mp_sh / fs_mp
        if sh_cov < min_fs_mp_sh_ratio:
            errors.append(
                f"FS∩MP∩SH/FS∩MP={sh_cov:.4f} < --min-fs-mp-sh-ratio {min_fs_mp_sh_ratio}"
            )
    if format_both_empty_pct > max_format_both_empty_pct:
        errors.append(
            f"format both-empty={format_both_empty_pct:.2f}% > "
            f"--max-format-both-empty-pct {max_format_both_empty_pct}"
        )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--feature-store-db", type=Path, default=None)
    ap.add_argument("--marketplace-db", type=Path, default=None)
    ap.add_argument("--sale-history-db", type=Path, default=None)
    ap.add_argument(
        "--format-audit",
        action="store_true",
        help="Report missing formats_json/format_desc on FS∩MP rows.",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if overlap/format metrics fail threshold flags below.",
    )
    ap.add_argument(
        "--min-mp-in-fs-ratio",
        type=float,
        default=0.90,
        help="Require FS∩MP/MP >= this (MP rows joinable to catalog). Default 0.90.",
    )
    ap.add_argument(
        "--min-fs-mp-sh-ratio",
        type=float,
        default=0.50,
        help="When sale_history exists: require FS∩MP∩SH/FS∩MP >= this. Default 0.50.",
    )
    ap.add_argument(
        "--max-format-both-empty-pct",
        type=float,
        default=1.0,
        help="Max %% of FS∩MP rows with both format fields empty. Default 1.0.",
    )
    args = ap.parse_args()

    root = _root()
    cfgp = args.config or (root / "configs" / "base.yaml")
    cfg: dict = {}
    if cfgp.is_file():
        cfg = yaml.safe_load(cfgp.read_text()) or {}
    paths = (cfg.get("vinyliq") or {}).get("paths") or {}

    fs = _resolve(root, args.feature_store_db, paths.get("feature_store_db", "data/feature_store.sqlite"))
    mp = _resolve(root, args.marketplace_db, paths.get("marketplace_db", "data/cache/marketplace_stats.sqlite"))
    sh = _resolve(root, args.sale_history_db, paths.get("sale_history_db", "data/cache/sale_history.sqlite"))

    print(f"feature_store: {fs}")
    print(f"marketplace:   {mp}")
    print(f"sale_history:  {sh}")

    if not fs.is_file():
        print(f"ERROR: feature_store missing: {fs}")
        return 1
    if not mp.is_file():
        print(f"ERROR: marketplace DB missing: {mp}")
        return 1

    conn = sqlite3.connect(":memory:")
    n_fs = n_mp = fs_mp = fs_mp_sh = 0
    format_both_empty_pct = 0.0
    try:
        conn.execute("ATTACH DATABASE ? AS fs", (str(fs),))
        conn.execute("ATTACH DATABASE ? AS mp", (str(mp),))
        has_sh = sh.is_file()
        if has_sh:
            conn.execute("ATTACH DATABASE ? AS sh", (str(sh),))

        n_fs = _count(conn, "SELECT COUNT(*) FROM fs.releases_features")
        n_mp = _count(conn, "SELECT COUNT(*) FROM mp.marketplace_stats")
        print(f"rows: FS={n_fs:,}  MP={n_mp:,}")

        fs_mp = _count(
            conn,
            "SELECT COUNT(*) FROM fs.releases_features f "
            "INNER JOIN mp.marketplace_stats m ON m.release_id = f.release_id",
        )
        print(f"overlap: FS∩MP={fs_mp:,}")
        if n_mp > 0:
            print(f"ratio: FS∩MP/MP={fs_mp / n_mp:.4f}")

        if has_sh:
            n_sh = _count(conn, "SELECT COUNT(*) FROM sh.sale_history_fetch_status")
            print(f"rows: SH(fetch_status)={n_sh:,}")
            fs_sh = _count(
                conn,
                "SELECT COUNT(*) FROM fs.releases_features f "
                "INNER JOIN sh.sale_history_fetch_status s ON s.release_id = f.release_id",
            )
            mp_sh = _count(
                conn,
                "SELECT COUNT(*) FROM mp.marketplace_stats m "
                "INNER JOIN sh.sale_history_fetch_status s ON s.release_id = m.release_id",
            )
            fs_mp_sh = _count(
                conn,
                "SELECT COUNT(*) FROM fs.releases_features f "
                "INNER JOIN mp.marketplace_stats m ON m.release_id = f.release_id "
                "INNER JOIN sh.sale_history_fetch_status s ON s.release_id = f.release_id",
            )
            print(f"overlap: FS∩SH={fs_sh:,}  MP∩SH={mp_sh:,}  FS∩MP∩SH={fs_mp_sh:,}")
            if fs_mp > 0:
                print(f"ratio: FS∩MP∩SH/FS∩MP={fs_mp_sh / fs_mp:.4f}")
        else:
            print("sale_history DB missing; SH overlaps skipped")
        if args.format_audit:
            _, format_both_empty_pct = _format_field_audit(conn)
    finally:
        conn.close()

    if args.check:
        errors = _check_thresholds(
            n_fs=n_fs,
            n_mp=n_mp,
            fs_mp=fs_mp,
            has_sh=sh.is_file(),
            fs_mp_sh=fs_mp_sh,
            format_both_empty_pct=format_both_empty_pct,
            min_mp_in_fs_ratio=args.min_mp_in_fs_ratio,
            min_fs_mp_sh_ratio=args.min_fs_mp_sh_ratio,
            max_format_both_empty_pct=args.max_format_both_empty_pct,
        )
        if errors:
            for msg in errors:
                print(f"CHECK FAILED: {msg}")
            return 1
        print("CHECK OK: all thresholds passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
