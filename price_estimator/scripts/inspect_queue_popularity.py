#!/usr/bin/env python3
"""
Sample release_ids from a queue file and show ``community_have`` /
``community_want`` from ``marketplace_stats.sqlite`` (plan §1b).

Indices are **numeric-ID line order** (blank lines and # comments skipped)—same ordering
as ``collect_marketplace_stats`` streaming reader.

Use this to verify a queue is really "popular-first" (early indices should have large
have+want if the file was built with combined/have/want sort).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_numeric_ids(path: Path):
    """Yield (k, release_id) where k is 0,1,2,... over non-empty numeric ID lines."""
    k = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()[0]
            if tok.isdigit():
                yield k, tok
                k += 1


def _count_numeric_ids(path: Path) -> int:
    return sum(1 for _ in _iter_numeric_ids(path))


def _fetch_at_indices(path: Path, indices: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    need = set(indices)
    for k, rid in _iter_numeric_ids(path):
        if k in need:
            out[k] = rid
            need.discard(k)
            if not need:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, required=True, help="One ID per line")
    ap.add_argument(
        "--marketplace-db",
        type=Path,
        default=None,
        help=(
            "marketplace_stats.sqlite "
            "(default: sibling cache/ next to feature_store from vinyliq.paths)"
        ),
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=15,
        help="How many consecutive IDs to show at head, each mid anchor, and tail",
    )
    ap.add_argument(
        "--mid-indices",
        type=str,
        default="100000,350000",
        help=(
            "Comma-separated **numeric-ID indices** (0=first ID) to sample from "
            "(e.g. 350000 right after a 350k popular cap)"
        ),
    )
    args = ap.parse_args()

    q = args.queue.expanduser().resolve()
    if not q.is_file():
        print(f"Not found: {q}", file=sys.stderr)
        return 1

    root = _root()
    mp_path = args.marketplace_db
    if mp_path is None:
        import yaml

        cfgp = root / "configs" / "base.yaml"
        if cfgp.is_file():
            with open(cfgp) as f:
                data = yaml.safe_load(f) or {}
            rel = (data.get("vinyliq") or {}).get("paths", {}).get(
                "marketplace_db", "data/cache/marketplace_stats.sqlite"
            )
            mp_path = Path(rel)
            if not mp_path.is_absolute():
                mp_path = root / mp_path
        else:
            mp_path = root / "data" / "cache" / "marketplace_stats.sqlite"
    else:
        mp_path = args.marketplace_db.expanduser().resolve()
        if not mp_path.is_absolute():
            mp_path = root / mp_path

    if not mp_path.is_file():
        print(f"marketplace_stats not found: {mp_path}", file=sys.stderr)
        return 1

    n_lines = _count_numeric_ids(q)
    print(f"queue: {q}")
    print(f"numeric release_id rows (0-based index 0..{n_lines - 1}): {n_lines}")
    print(f"marketplace_stats: {mp_path}\n")

    mconn = sqlite3.connect(str(mp_path))
    mconn.row_factory = sqlite3.Row

    def community_for(rid: str) -> tuple[int | None, int | None]:
        cur = mconn.execute(
            "SELECT community_have, community_want FROM marketplace_stats "
            "WHERE release_id = ?",
            (rid,),
        )
        r = cur.fetchone()
        if not r:
            return None, None
        h = r["community_have"]
        w = r["community_want"]
        try:
            hi = int(h) if h is not None else None
        except (TypeError, ValueError):
            hi = None
        try:
            wi = int(w) if w is not None else None
        except (TypeError, ValueError):
            wi = None
        return hi, wi

    n = max(1, args.sample)
    mids: list[int] = []
    for part in args.mid_indices.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            mids.append(int(part))
        except ValueError:
            print(f"Ignoring bad --mid-indices entry: {part!r}", file=sys.stderr)

    want: set[int] = set()
    for j in range(min(n, n_lines)):
        want.add(j)
    for off in mids:
        for j in range(n):
            idx = off + j
            if idx < n_lines:
                want.add(idx)
    for j in range(min(n, n_lines)):
        want.add(max(0, n_lines - 1 - j))

    id_map = _fetch_at_indices(q, want)

    def show(label: str, indices: list[int]) -> None:
        print(f"--- {label} ---")
        any_row = False
        for idx in indices:
            if idx >= n_lines:
                continue
            rid = id_map.get(idx)
            if rid is None:
                continue
            h, w = community_for(rid)
            hs = "?" if h is None else str(h)
            ws = "?" if w is None else str(w)
            comb = "?" if h is None or w is None else str(h + w)
            print(f"  idx={idx:>8}  id={rid:>12}  have={hs:>8}  want={ws:>8}  sum={comb:>8}")
            any_row = True
        if not any_row:
            print("(no rows)")

    show(
        "HEAD (hottest if queue is combined/have/want-sorted)",
        list(range(min(n, n_lines))),
    )
    for off in mids:
        idxs = [off + j for j in range(n) if off + j < n_lines]
        if idxs:
            show(f"FROM index {off} ({n} rows)", idxs)
    tail_idxs = [max(0, n_lines - 1 - j) for j in range(min(n, n_lines))]
    show("TAIL", sorted(set(tail_idxs)))

    mconn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
