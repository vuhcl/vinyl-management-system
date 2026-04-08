#!/usr/bin/env python3
"""
Build a deduped ``release_id`` queue for ``collect_marketplace_stats.py``.

Merges:

1. **Primary block** (size *N* = ``--primary-limit``): order by ``--rank-by``.
   Default **proxy** = catalog-based score (master fan-out + primary-artist catalog
   mass, then year, label_tier, release_id). Use **combined** / **have** / **want**
   only when ``releases_features`` has real Discogs **community** counts (e.g. API
   ingest); with dump-only data those columns are usually all zero, so ordering
   collapses to ``release_id``.

   Optional second block: ``--extra-limit`` more IDs (for **proxy**, same catalog
   order skipping duplicates). For community sorts, the extra block uses the
   complement sort (want vs have).

2. **Stratified** — Up to *K* IDs per bucket. Default **random** (deterministic hash).
   **proxy** = highest catalog score per bucket. **community** (alias **popularity**)
   = highest have+want per bucket (useless if counts are all zero).

3. **Dedup** — Primary list first, then stratified IDs not already present.
   Optional ``--max-total`` truncates the final list.

Requires SQLite 3.25+ (window functions) for stratified passes; JSON1 for **proxy**.

**Various-artist compilations** (Discogs Various id **194**, or primary artist name
containing ``various``) and **Unknown Artist** as the primary credit are
**excluded** from the output queue and from proxy master/artist counts so
samplers are not dominated by compilations or placeholders.

**File / digital** releases (Discogs format **File** in ``formats_json`` or
``format_desc``) are **excluded**. **Vinyl media** (LP, 7\"/10\"/12\", format
**vinyl**, etc.) are identified for sorting and optional **target share**
(``--target-vinyl-fraction``, default **0.7**): each primary block, extra block,
and stratified bucket aims for that fraction vinyl vs other physical formats;
use **0** to disable quota (tie-break only).

**Artist diversity:** ``--max-per-primary-artist`` (default **5**, **0** = no cap)
limits how many releases each primary Discogs artist can contribute to the
proxy **head** and, for ``--stratify-order proxy``, within each stratification
bucket—so one mega-catalog artist cannot occupy the whole queue.

Example:

  PYTHONPATH=. python price_estimator/scripts/build_stats_collection_queue.py \\
      --db price_estimator/data/feature_store.sqlite \\
      --out price_estimator/data/raw/collection_queue.txt \\
      --rank-by proxy \\
      --primary-limit 250000 \\
      --extra-limit 50000 \\
      --stratify-per-bucket 40 \\
      --stratify-by decade_genre \\
      --stratify-order proxy \\
      --seed 42 \\
      --max-total 200000

Then feed ``--release-ids`` to ``collect_marketplace_stats.py`` with ``--resume``.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections.abc import Callable, Iterator
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def max_community_sum(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT MAX(COALESCE(have_count,0) + COALESCE(want_count,0)) "
            "FROM releases_features"
        ).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def warn_if_community_sort_useless(
    db_path: Path,
    *,
    rank_by: str,
    stratify_order: str,
) -> None:
    uses_community = rank_by in ("combined", "have", "want") or stratify_order in (
        "community",
        "popularity",
    )
    if not uses_community:
        return
    mx = max_community_sum(db_path)
    if mx > 0:
        return
    print(
        "Warning: community ordering (have/want) selected but "
        "MAX(have_count+want_count) is 0 in this feature store — "
        "order matches release_id only. Use --rank-by proxy for "
        "catalog-based ranking.",
        file=sys.stderr,
    )


def iter_catalog_proxy_release_ids(
    db_path: Path,
    w_master: float,
    w_artist: float,
) -> Iterator[str]:
    for rid, _aid, _v in iter_catalog_proxy_release_rows(
        db_path, w_master, w_artist
    ):
        yield rid


def iter_catalog_proxy_release_rows(
    db_path: Path,
    w_master: float,
    w_artist: float,
) -> Iterator[tuple[str, str, int]]:
    """Yield (release_id, primary_artist_id_text, vinyl_flag 0/1) in proxy order."""
    from price_estimator.src.catalog_proxy import (
        sql_select_release_id_and_primary_artist_ordered_by_catalog_proxy,
    )

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            sql_select_release_id_and_primary_artist_ordered_by_catalog_proxy(),
            (w_master, w_artist),
        )
        while True:
            row = cur.fetchone()
            if row is None:
                break
            aid = str(row[1]) if row[1] is not None else ""
            v = int(row[2]) if row[2] is not None else 0
            yield str(row[0]), aid, v
    finally:
        conn.close()


def _take_vinyl_quota_block(
    iter_rows_factory: Callable[[], Iterator[tuple[str, str, int]]],
    *,
    block_limit: int,
    max_per_primary_artist: int,
    seen: set[str],
    per_artist: dict[str, int],
    skip_if_seen: bool,
    target_vinyl_fraction: float | None,
    out: list[str],
) -> None:
    """
    Consume rows from *iter_rows_factory* (release_id, primary_artist_id, vinyl).

    Appends up to *block_limit* ids to *out* while respecting optional vinyl
    share, artist cap, and dedupe *seen*. A second pass relaxes quota if the
    first pass ends before *block_limit*.
    """
    limit = max(0, int(block_limit))
    if limit <= 0:
        return
    cap = max(0, int(max_per_primary_artist))
    use_quota = (
        target_vinyl_fraction is not None
        and 0 < float(target_vinyl_fraction) < 1
    )
    need_v = int(round(limit * float(target_vinyl_fraction))) if use_quota else 0
    need_nv = limit - need_v if use_quota else 0
    got_v = 0
    got_nv = 0
    appended = 0

    def _one_pass(*, strict_quota: bool) -> None:
        nonlocal got_v, got_nv, appended
        for rid, aid, v in iter_rows_factory():
            if appended >= limit:
                break
            if skip_if_seen and rid in seen:
                continue
            if cap > 0 and per_artist.get(aid, 0) >= cap:
                continue
            is_v = int(v) == 1
            if strict_quota and use_quota:
                if got_v < need_v or got_nv < need_nv:
                    if is_v:
                        allow = (got_v < need_v) or (got_nv >= need_nv)
                    else:
                        allow = (got_nv < need_nv) or (got_v >= need_v)
                    if not allow:
                        continue
                    got_v += is_v
                    got_nv += not is_v
            out.append(rid)
            seen.add(rid)
            per_artist[aid] = per_artist.get(aid, 0) + 1
            appended += 1

    if use_quota:
        _one_pass(strict_quota=True)
        if appended < limit:
            _one_pass(strict_quota=False)
    else:
        _one_pass(strict_quota=False)


def _collect_ranked_ids_proxy(
    db_path: Path,
    *,
    primary_limit: int,
    extra_limit: int,
    w_master: float,
    w_artist: float,
    max_per_primary_artist: int,
    target_vinyl_fraction: float | None = None,
) -> list[str]:
    """
    Primary block then optional extra IDs in proxy order.

    If ``max_per_primary_artist`` > 0, skip a row when that primary artist id
    is already at the cap (shared across primary + extra). **0** = no cap.

    ``target_vinyl_fraction`` in (0, 1): target vinyl share within each block;
    **None** or outside that range: no quota (proxy order + tie-break only).
    """
    out: list[str] = []
    seen: set[str] = set()
    per_artist: dict[str, int] = {}
    cap = max(0, int(max_per_primary_artist))

    if max(0, primary_limit) > 0:
        _take_vinyl_quota_block(
            lambda: iter_catalog_proxy_release_rows(
                db_path, w_master, w_artist
            ),
            block_limit=primary_limit,
            max_per_primary_artist=cap,
            seen=seen,
            per_artist=per_artist,
            skip_if_seen=False,
            target_vinyl_fraction=target_vinyl_fraction,
            out=out,
        )
    if max(0, extra_limit) > 0:
        _take_vinyl_quota_block(
            lambda: iter_catalog_proxy_release_rows(
                db_path, w_master, w_artist
            ),
            block_limit=extra_limit,
            max_per_primary_artist=cap,
            seen=seen,
            per_artist=per_artist,
            skip_if_seen=True,
            target_vinyl_fraction=target_vinyl_fraction,
            out=out,
        )
    return out


def collect_ranked_ids(
    db_path: Path,
    *,
    primary_limit: int,
    extra_limit: int,
    rank_by: str = "proxy",
    w_master: float = 1.0,
    w_artist: float = 1.0,
    max_per_primary_artist: int = 5,
    target_vinyl_fraction: float | None = None,
) -> list[str]:
    if rank_by == "proxy":
        return _collect_ranked_ids_proxy(
            db_path,
            primary_limit=primary_limit,
            extra_limit=extra_limit,
            w_master=w_master,
            w_artist=w_artist,
            max_per_primary_artist=max_per_primary_artist,
            target_vinyl_fraction=target_vinyl_fraction,
        )

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
    if rank_by not in primary_map:
        raise ValueError(
            f"rank_by must be one of proxy,{tuple(primary_map)}, "
            f"got {rank_by!r}"
        )
    primary_sort = primary_map[rank_by]
    secondary_sort = (
        "have_count" if rank_by == "want" else "want_count"
    )

    store = FeatureStoreDB(db_path)
    out: list[str] = []
    seen: set[str] = set()
    per_artist: dict[str, int] = {}

    if primary_limit > 0:
        _take_vinyl_quota_block(
            lambda: (
                (rid, "", v)
                for rid, v in store.iter_community_release_rows(
                    sort_by=primary_sort,
                    min_have=0,
                    min_want=0,
                    exclude_various_artists=True,
                    exclude_file_formats=True,
                    prefer_vinyl_tiebreak=True,
                )
            ),
            block_limit=primary_limit,
            max_per_primary_artist=0,
            seen=seen,
            per_artist=per_artist,
            skip_if_seen=False,
            target_vinyl_fraction=target_vinyl_fraction,
            out=out,
        )

    if extra_limit > 0:
        _take_vinyl_quota_block(
            lambda: (
                (rid, "", v)
                for rid, v in store.iter_community_release_rows(
                    sort_by=secondary_sort,
                    min_have=0,
                    min_want=0,
                    exclude_various_artists=True,
                    exclude_file_formats=True,
                    prefer_vinyl_tiebreak=True,
                )
            ),
            block_limit=extra_limit,
            max_per_primary_artist=0,
            seen=seen,
            per_artist=per_artist,
            skip_if_seen=True,
            target_vinyl_fraction=target_vinyl_fraction,
            out=out,
        )

    return out


def collect_stratified_ids(
    db_path: Path,
    *,
    per_bucket: int,
    stratify_by: str,
    seed: int,
    order: str = "random",
    w_master: float = 1.0,
    w_artist: float = 1.0,
    max_per_primary_artist_per_bucket: int = 0,
    target_vinyl_fraction: float | None = None,
) -> list[str]:
    """
    Up to ``per_bucket`` rows per partition.

    ``order``:
      ``random`` — deterministic hash of ``release_id`` and ``seed``;
      ``community`` — highest have+want per bucket first;
      ``proxy`` — highest catalog proxy score per bucket.

    Optional ``target_vinyl_fraction`` in (0, 1): split each bucket into that
    share vinyl vs non-vinyl (same ordering within each stratum).
    """
    if per_bucket <= 0:
        return []

    if order == "popularity":
        order = "community"

    conn = sqlite3.connect(str(db_path))
    try:
        from price_estimator.src.catalog_proxy import (
            sql_exclude_file_format_releases,
            sql_exclude_various_primary_artist,
            sql_vinyl_preference_key,
            stratified_vinyl_bucket_counts,
        )

        queue_row_filter = (
            f"({sql_exclude_various_primary_artist('artists_json')}) AND "
            f"({sql_exclude_file_format_releases('formats_json', 'format_desc')})"
        )
        _vinyl_plain = sql_vinyl_preference_key("formats_json", "format_desc")

        split: tuple[int, int] | None = None
        if target_vinyl_fraction is not None and 0 < float(
            target_vinyl_fraction
        ) < 1:
            split = stratified_vinyl_bucket_counts(
                per_bucket, float(target_vinyl_fraction)
            )

        if stratify_by == "decade":
            partition_plain = "decade"
            partition_inner = "decade"
        else:
            partition_plain = "decade, COALESCE(genre, '')"
            partition_inner = "decade, _g"

        if order == "community":
            if split is not None:
                kv, kn = split
                sql = f"""
WITH inner AS (
  SELECT release_id,
         decade,
         COALESCE(genre, '') AS _g,
         CAST(({_vinyl_plain}) AS INTEGER) AS _v,
         (COALESCE(have_count, 0) + COALESCE(want_count, 0)) AS pop
  FROM releases_features
  WHERE {queue_row_filter}
),
vin AS (
  SELECT release_id FROM (
    SELECT release_id,
           row_number() OVER (
             PARTITION BY {partition_inner}
             ORDER BY pop DESC, _v DESC, release_id ASC
           ) AS rn
    FROM inner WHERE _v = 1
  ) WHERE rn <= ?
),
non AS (
  SELECT release_id FROM (
    SELECT release_id,
           row_number() OVER (
             PARTITION BY {partition_inner}
             ORDER BY pop DESC, _v DESC, release_id ASC
           ) AS rn
    FROM inner WHERE _v = 0
  ) WHERE rn <= ?
)
SELECT release_id FROM vin
UNION ALL
SELECT release_id FROM non
"""
                params = (kv, kn)
            else:
                inner_order = (
                    "(COALESCE(have_count, 0) + COALESCE(want_count, 0)) DESC, "
                    f"{_vinyl_plain} DESC, "
                    "release_id ASC"
                )
                sql = f"""
                SELECT release_id FROM (
                  SELECT release_id,
                         row_number() OVER (
                           PARTITION BY {partition_plain}
                           ORDER BY {inner_order}
                         ) AS rn
                  FROM releases_features
                  WHERE {queue_row_filter}
                )
                WHERE rn <= ?
                """
                params = (per_bucket,)
        elif order == "proxy":
            from price_estimator.src.catalog_proxy import (
                sql_stratified_release_ids_catalog_proxy,
            )

            mpp = max(0, int(max_per_primary_artist_per_bucket))
            sql = sql_stratified_release_ids_catalog_proxy(
                stratify_by,
                max_per_primary_artist_per_bucket=mpp,
                vinyl_bucket_split=split,
            )
            if mpp > 0:
                if split is not None:
                    kv, kn = split
                    params = (w_master, w_artist, mpp, kv, kn)
                else:
                    params = (w_master, w_artist, mpp, per_bucket)
            else:
                if split is not None:
                    kv, kn = split
                    params = (w_master, w_artist, kv, kn)
                else:
                    params = (w_master, w_artist, per_bucket)
        else:
            if split is not None:
                kv, kn = split
                sql = f"""
WITH inner AS (
  SELECT release_id,
         decade,
         COALESCE(genre, '') AS _g,
         CAST(({_vinyl_plain}) AS INTEGER) AS _v,
         abs((CAST(release_id AS INTEGER) * 7919 + ?) % 2147483647) AS h
  FROM releases_features
  WHERE {queue_row_filter}
),
vin AS (
  SELECT release_id FROM (
    SELECT release_id,
           row_number() OVER (
             PARTITION BY {partition_inner}
             ORDER BY h
           ) AS rn
    FROM inner WHERE _v = 1
  ) WHERE rn <= ?
),
non AS (
  SELECT release_id FROM (
    SELECT release_id,
           row_number() OVER (
             PARTITION BY {partition_inner}
             ORDER BY h
           ) AS rn
    FROM inner WHERE _v = 0
  ) WHERE rn <= ?
)
SELECT release_id FROM vin
UNION ALL
SELECT release_id FROM non
"""
                params = (seed, kv, kn)
            else:
                sql = f"""
                SELECT release_id FROM (
                  SELECT release_id,
                         row_number() OVER (
                           PARTITION BY {partition_plain}
                           ORDER BY abs(
                             (CAST(release_id AS INTEGER) * 7919 + ?) % 2147483647
                           )
                         ) AS rn
                  FROM releases_features
                  WHERE {queue_row_filter}
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
    head_ids: list[str], stratified: list[str], max_total: int
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for rid in head_ids:
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
        description=(
            "Build merged catalog-proxy (or community) + stratified release_id "
            "queue for stats collection"
        ),
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
        "--primary-limit",
        type=int,
        default=0,
        help=(
            "Head of queue: this many release_ids from the top of --rank-by order "
            "(0=skip)"
        ),
    )
    p.add_argument(
        "--extra-limit",
        type=int,
        default=0,
        help=(
            "After the primary block: for --rank-by proxy, more IDs in the same "
            "catalog order (skipping duplicates). For combined/have/want, IDs "
            "from the complement community sort."
        ),
    )
    p.add_argument(
        "--rank-by",
        choices=("proxy", "combined", "have", "want"),
        default="proxy",
        help=(
            "proxy=catalog proxy score (default); combined/have/want=community "
            "columns in DB (usually zero after dump ingest)"
        ),
    )
    p.add_argument(
        "--proxy-weight-master",
        type=float,
        default=1.0,
        help="Weight for releases per master_id (proxy sort only, default 1)",
    )
    p.add_argument(
        "--proxy-weight-artist",
        type=float,
        default=1.0,
        help="Weight for releases per primary artist id (proxy sort only, default 1)",
    )
    p.add_argument(
        "--max-per-primary-artist",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Proxy only: max releases per primary Discogs artist in the head+extra "
            "blocks, and per decade[/genre] bucket when using --stratify-order proxy. "
            "0 = no limit (mega-artists can dominate)."
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
        choices=("random", "community", "proxy", "popularity"),
        default="random",
        help=(
            "Within each bucket: random (default), proxy (catalog score), "
            "community or popularity (have+want; alias popularity=community)"
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
    p.add_argument(
        "--target-vinyl-fraction",
        type=float,
        default=0.7,
        metavar="F",
        help=(
            "Target vinyl-medium share (0–1) in each primary/extra block and "
            "each stratified bucket; 0 disables quota (sort tie-break only). "
            "Default 0.7"
        ),
    )
    args = p.parse_args()

    strat_order = args.stratify_order
    if strat_order == "popularity":
        strat_order = "community"

    root = _root()
    db_path = args.db or (root / "data" / "feature_store.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path

    if not db_path.is_file():
        print(f"Feature store not found: {db_path}", file=sys.stderr)
        return 1

    if (
        args.primary_limit <= 0
        and args.extra_limit <= 0
        and args.stratify_per_bucket <= 0
    ):
        print(
            "Need at least one of --primary-limit, --extra-limit, "
            "or --stratify-per-bucket > 0.",
            file=sys.stderr,
        )
        return 1

    warn_if_community_sort_useless(
        db_path,
        rank_by=args.rank_by,
        stratify_order=strat_order,
    )

    mpp = max(0, int(args.max_per_primary_artist))
    vf = float(args.target_vinyl_fraction)
    vinyl_quota: float | None = None if vf <= 0 or vf >= 1 else vf
    head_ids = collect_ranked_ids(
        db_path,
        primary_limit=max(0, args.primary_limit),
        extra_limit=max(0, args.extra_limit),
        rank_by=args.rank_by,
        w_master=float(args.proxy_weight_master),
        w_artist=float(args.proxy_weight_artist),
        max_per_primary_artist=mpp if args.rank_by == "proxy" else 0,
        target_vinyl_fraction=vinyl_quota,
    )
    strat_mpp = mpp if strat_order == "proxy" else 0
    strat = collect_stratified_ids(
        db_path,
        per_bucket=max(0, args.stratify_per_bucket),
        stratify_by=args.stratify_by,
        seed=args.seed,
        order=strat_order,
        w_master=float(args.proxy_weight_master),
        w_artist=float(args.proxy_weight_artist),
        max_per_primary_artist_per_bucket=strat_mpp,
        target_vinyl_fraction=vinyl_quota,
    )
    merged = merge_dedupe(
        head_ids,
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

    head_set = set(head_ids)
    n_from_strat_only = sum(1 for r in merged if r not in head_set)
    if (
        args.rank_by == "proxy"
        and mpp > 0
        and args.primary_limit > 0
        and len(head_ids) < args.primary_limit
    ):
        print(
            f"Note: head block has {len(head_ids)} ids (< --primary-limit "
            f"{args.primary_limit}); --max-per-primary-artist {mpp} may have "
            "exhausted eligible rows before filling the cap.",
            file=sys.stderr,
        )
    print(
        f"Wrote {len(merged)} release_ids → {args.out} "
        f"(head_block={len(head_ids)}, stratified_sql_rows={len(strat)}, "
        f"ids_in_output_not_in_head_block={n_from_strat_only})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
