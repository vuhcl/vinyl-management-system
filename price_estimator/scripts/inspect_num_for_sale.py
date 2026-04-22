#!/usr/bin/env python3
"""num_for_sale distribution from marketplace_stats.sqlite (optional training filter)."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml


def _default_db() -> Path:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "base.yaml"
    if cfg.is_file():
        with open(cfg) as f:
            data = yaml.safe_load(f) or {}
        p = (data.get("vinyliq") or {}).get("paths") or {}
        rel = p.get("marketplace_db", "data/cache/marketplace_stats.sqlite")
        out = Path(rel)
        if not out.is_absolute():
            out = root / out
        return out
    return root / "data" / "cache" / "marketplace_stats.sqlite"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to marketplace_stats.sqlite (default: from vinyliq.paths)",
    )
    ap.add_argument(
        "--training-rows",
        action="store_true",
        help=(
            "Only rows with release_lowest_price NOT NULL AND >0 "
            "(same filter as training labels)"
        ),
    )
    args = ap.parse_args()
    db = args.db.expanduser().resolve() if args.db else _default_db()
    if not db.is_file():
        print(f"No database at {db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db))
    where = "1=1"
    if args.training_rows:
        where = "release_lowest_price IS NOT NULL AND release_lowest_price > 0"
    sql = f"SELECT num_for_sale FROM marketplace_stats WHERE {where}"
    cur = conn.execute(sql)
    raw = [r[0] for r in cur.fetchall()]
    conn.close()

    n = len(raw)
    if n == 0:
        print("No rows.")
        return 0

    x = np.array([int(v or 0) for v in raw], dtype=np.int64)
    zeros = int(np.sum(x == 0))
    print(f"database: {db}")
    flt = "training (release_lowest_price>0)" if args.training_rows else "all rows"
    print(f"filter: {flt}")
    print(f"n={n}")
    print(f"num_for_sale == 0: {zeros} ({100.0 * zeros / n:.2f}%)")
    print(
        f"min={int(x.min())}  max={int(x.max())}  mean={float(x.mean()):.2f}  "
        f"median={float(np.median(x)):.1f}"
    )
    for p in (50, 75, 90, 95, 99):
        print(f"  P{p}: {float(np.percentile(x, p)):.1f}")
    # log-scale bins for non-zero
    pos = x[x > 0]
    if len(pos):
        lp = np.log10(pos.astype(np.float64))
        p50 = float(np.median(lp))
        p90 = float(np.percentile(lp, 90))
        print(f"Among num_for_sale > 0 (n={len(pos)}): log10 P50={p50:.3f} P90={p90:.3f}")
    # coarse histogram
    edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 10**9]
    print("histogram [lo, hi) count pct:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = int(np.sum((x >= lo) & (x < hi)))
        hi_s = f"{hi:>4}" if hi < 10**8 else " inf"
        pct = 100.0 * c / n
        print(f"  [{lo:>4},{hi_s}): {c:>7}  {pct:>6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
