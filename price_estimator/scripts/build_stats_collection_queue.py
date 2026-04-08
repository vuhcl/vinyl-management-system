#!/usr/bin/env python3
"""
Build a deduped ``release_id`` queue for ``collect_marketplace_stats.py``.

Merges:

1. **Popularity** — Primary block (size *N* = ``--popular-have-limit``): order by
   ``--popular-by`` (default **combined** = ``have_count + want_count`` desc, so
   overall community demand leads). Then up to *M* distinct IDs from a **secondary**
   sort (``want_count`` if primary is ``have`` or ``combined``, else ``have_count``)
   so high-want or high-have tails still get coverage.

2. **Stratified** — Up to *K* IDs per bucket. Default order is **deterministic**
   pseudo-random (``--stratify-order random``). Use ``--stratify-order popularity``
   to take the *K* most popular (have+want) rows per bucket instead.

3. **Dedup** — Popular list first, then stratified IDs not already present.
   Optional ``--max-total`` truncates the final list.

The stratified pass uses a single window query (SQLite 3.25+); it scans the full
``releases_features`` table once.

Example (≈70% / 30% style mix via explicit caps):

  PYTHONPATH=. python price_estimator/scripts/build_stats_collection_queue.py \\
      --db price_estimator/data/feature_store.sqlite \\
      --out price_estimator/data/raw/collection_queue.txt \\
      --popular-by combined \\
      --popular-have-limit 250000 \\
      --popular-want-limit 250000 \\
      --stratify-per-bucket 40 \\
      --stratify-by decade_genre \\
      --stratify-order popularity \\
      --seed 42 \\
      --max-total 200000

Then feed ``--release-ids`` to ``collect_marketplace_stats.py`` with ``--resume``.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def collect_popular_ids(
    db_path: Path,
    *,
    have_limit: int,
    want_limit: int,
    popular_by: str = "combined",
) -> list[str]:
    """
    Primary block: first ``have_limit`` IDs by ``popular_by`` sort.

    Secondary: up to ``want_limit`` new IDs from the complement sort
    (want-order if primary is have or combined; have-order if primary is want).
    """
    try:
        from price_estimator.src.storage.feature_store import FeatureStoreDB
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        raise SystemExit(1) from None

    primary_map = {
        "have": "have_count",
        "want": "want_count",
        "combined": "popularity",
    }
    if popular_by not in primary_map:
        raise ValueError(f"popular_by must be one of {tuple(primary_map)}, got {popular_by!r}")
    primary_sort = primary_map[popular_by]
    secondary_sort = (
        "have_count" if popular_by == "want" else "want_count"
    )

    store = FeatureStoreDB(db_path)
    out: list[str] = []
    seen: set[str] = set()

    if have_limit > 0:
        for i, rid in enumerate(
            store.iter_release_ids(sort_by=primary_sort, min_have=0, min_want=0),
        ):
            if i >= have_limit:
                break
            seen.add(rid)
            out.append(rid)

    if want_limit > 0:
        n_sec = 0
        for rid in store.iter_release_ids(
            sort_by=secondary_sort, min_have=0, min_want=0
        ):
            if rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
            n_sec += 1
            if n_sec >= want_limit:
                break

    return out


def collect_stratified_ids(
    db_path: Path,
    *,
    per_bucket: int,
    stratify_by: str,
    seed: int,
    order: str = "random",
) -> list[str]:
    """
    Up to ``per_bucket`` rows per partition.

    ``order``:
      ``random`` — deterministic hash of ``release_id`` and ``seed`` (reproducible);
      ``popularity`` — highest ``have_count + want_count`` per bucket first.
    """
    if per_bucket <= 0:
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        if stratify_by == "decade":
            partition = "decade"
        else:
            partition = "decade, COALESCE(genre, '')"

        if order == "popularity":
            inner_order = (
                "(COALESCE(have_count, 0) + COALESCE(want_count, 0)) DESC, "
                "release_id ASC"
            )
            sql = f"""
                SELECT release_id FROM (
                  SELECT release_id,
                         row_number() OVER (
                           PARTITION BY {partition}
                           ORDER BY {inner_order}
                         ) AS rn
                  FROM releases_features
                )
                WHERE rn <= ?
                """
            params: tuple[int, ...] = (per_bucket,)
        else:
            sql = f"""
                SELECT release_id FROM (
                  SELECT release_id,
                         row_number() OVER (
                           PARTITION BY {partition}
                           ORDER BY abs(
                             (CAST(release_id AS INTEGER) * 7919 + ?) % 2147483647
                           )
                         ) AS rn
                  FROM releases_features
                )
                WHERE rn <= ?
                """
            params = (seed, per_bucket)

        try:
            cur = conn.execute(sql, params)
            return [str(row[0]) for row in cur.fetchall()]
        except sqlite3.OperationalError as e:
            print(
                "Stratified sampling failed (need SQLite 3.25+ for window functions): "
                f"{e}",
                file=sys.stderr,
            )
            return []
    finally:
        conn.close()


def merge_dedupe(
    popular: list[str], stratified: list[str], max_total: int
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for rid in popular:
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
        if max_total > 0 and len(out) >= max_total:
            return out

    for rid in stratified:
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
        if max_total > 0 and len(out) >= max_total:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build merged popularity + stratified release_id queue for stats collection",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="feature_store.sqlite (default: price_estimator/data/feature_store.sqlite)",
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Output file, one ID per line"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for stratified ordering and optional --shuffle-final",
    )
    p.add_argument(
        "--popular-have-limit",
        type=int,
        default=0,
        help="Take this many release_ids from the top of have_count order (0=skip)",
    )
    p.add_argument(
        "--popular-want-limit",
        type=int,
        default=0,
        help=(
            "After the primary block, add up to this many new IDs from the "
            "secondary sort (want_count if primary is have/combined, else have_count)"
        ),
    )
    p.add_argument(
        "--popular-by",
        choices=("combined", "have", "want"),
        default="combined",
        help=(
            "Primary popularity ordering for the first block (size --popular-have-limit): "
            "combined = have+want desc (default), have = have_count desc, want = want_count desc"
        ),
    )
    p.add_argument(
        "--stratify-per-bucket",
        type=int,
        default=0,
        help="Max IDs per bucket from stratified pass (0=skip stratified)",
    )
    p.add_argument(
        "--stratify-by",
        choices=("decade_genre", "decade"),
        default="decade_genre",
        help="Partition for stratified sampling",
    )
    p.add_argument(
        "--stratify-order",
        choices=("random", "popularity"),
        default="random",
        help=(
            "Within each bucket, pick rows by deterministic random tie-break (default) "
            "or by have+want descending (popular-first coverage)"
        ),
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Cap final deduped list length (0=no cap)",
    )
    p.add_argument(
        "--shuffle-final",
        action="store_true",
        help="Shuffle merged IDs before write (same seed → same permutation)",
    )
    args = p.parse_args()

    root = _root()
    db_path = args.db or (root / "data" / "feature_store.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path

    if not db_path.is_file():
        print(f"Feature store not found: {db_path}", file=sys.stderr)
        return 1

    if (
        args.popular_have_limit <= 0
        and args.popular_want_limit <= 0
        and args.stratify_per_bucket <= 0
    ):
        print(
            "Need at least one of --popular-have-limit, --popular-want-limit, "
            "or --stratify-per-bucket > 0.",
            file=sys.stderr,
        )
        return 1

    popular = collect_popular_ids(
        db_path,
        have_limit=max(0, args.popular_have_limit),
        want_limit=max(0, args.popular_want_limit),
        popular_by=args.popular_by,
    )
    strat = collect_stratified_ids(
        db_path,
        per_bucket=max(0, args.stratify_per_bucket),
        stratify_by=args.stratify_by,
        seed=args.seed,
        order=args.stratify_order,
    )
    merged = merge_dedupe(
        popular,
        strat,
        max_total=max(0, args.max_total),
    )

    if args.shuffle_final:
        rng = random.Random(args.seed)
        rng.shuffle(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rid in merged:
            f.write(rid + "\n")

    pop_set = set(popular)
    n_from_strat_only = sum(1 for r in merged if r not in pop_set)
    print(
        f"Wrote {len(merged)} release_ids → {args.out} "
        f"(popular_block={len(popular)}, stratified_sql_rows={len(strat)}, "
        f"ids_in_output_not_in_popular_block={n_from_strat_only})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
