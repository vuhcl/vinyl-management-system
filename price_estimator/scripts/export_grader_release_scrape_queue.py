#!/usr/bin/env python3
"""
Export ``release_id`` lines for ``collect_discogs_release_listings_botasaurus``.

Joins ``releases_features`` (catalog year, format) with ``marketplace_stats``
(depth, ``release_lowest_price``). Rows must match **physical vinyl** (LP /
7\", 10\", 12\", / other vinyl hints in ``format_desc`` / ``formats_json``)
via ``catalog_proxy.sql_vinyl_format_rank`` — same notion as catalog-proxy
ordering.

Default ``--sort-by listing_floor`` ranks **higher** ``release_lowest_price``
first (expensive listings tend to carry more grade-conditioned signal), then
by listing depth and catalog year. Use ``--sort-by depth`` for the legacy
``num_for_sale``-first order.

By default, ``--vintage-before-year 2000`` lists **catalog year before that year**
(and rows with missing year) **before** newer releases, so the queue is not
dominated by recent reissues; within each era, price and depth ordering apply.
Pass ``0`` to disable that vintage-first key.

Merge with a manual list (e.g. ``release_scrape_queue_auto.txt`` +
``release_scrape_queue_manual.txt``) using ordered dedupe (see module doc).

Releases missing ``num_for_sale`` on ``marketplace_stats`` are excluded; enrich
via ``collect_marketplace_stats.py`` / ``GET /releases``, or add IDs by hand.

**Ctrl+C:** A SQLite ``progress_handler`` checks for interrupt so long queries
can exit instead of wedging until completion.
"""
from __future__ import annotations

import argparse
import signal
import sqlite3
import sys
from pathlib import Path

from price_estimator.src.catalog_proxy import sql_vinyl_format_rank


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export release_id queue for Discogs release marketplace scraping",
    )
    parser.add_argument(
        "--feature-db",
        type=Path,
        default=None,
        help="feature_store.sqlite (default under repo: price_estimator/data/…)",
    )
    parser.add_argument(
        "--marketplace-db",
        type=Path,
        default=None,
        help="marketplace_stats.sqlite (default: feature-db …/cache/…)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output text file (one release_id per line)",
    )
    parser.add_argument(
        "--min-num-for-sale",
        type=int,
        default=1,
        metavar="N",
        help="Min COALESCE(num_for_sale, release_num_for_sale, 0) (default 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max IDs to write (0 = all)",
    )
    parser.add_argument(
        "--sort-by",
        choices=("listing_floor", "depth"),
        default="listing_floor",
        help=(
            "listing_floor: higher release_lowest_price first, then depth (default); "
            "depth: num_for_sale first, then listing floor (legacy)"
        ),
    )
    parser.add_argument(
        "--vintage-before-year",
        type=int,
        default=2000,
        metavar="Y",
        help=(
            "Prefer f.year < Y (and missing year) before newer releases (default 2000); "
            "use 0 to sort by price/depth only"
        ),
    )
    parser.add_argument(
        "--all-formats",
        action="store_true",
        help="Do not require vinyl (LP/7\"/10\"/12\"/vinyl); include CD/file-like rows",
    )
    args = parser.parse_args()

    root = _root().parent
    fs = args.feature_db or (root / "price_estimator" / "data" / "feature_store.sqlite")
    if not fs.is_absolute():
        fs = root / fs
    mp = args.marketplace_db or (fs.parent / "cache" / "marketplace_stats.sqlite")
    if not mp.is_absolute():
        mp = root / mp

    if not fs.is_file():
        print(f"feature_store not found: {fs}", file=sys.stderr)
        return 1
    if not mp.is_file():
        print(f"marketplace_stats not found: {mp}", file=sys.stderr)
        return 1

    vintage_y = int(args.vintage_before_year)
    if vintage_y != 0 and not (1800 <= vintage_y <= 2100):
        print(
            "--vintage-before-year must be 0 (disabled) or in [1800, 2100]",
            file=sys.stderr,
        )
        return 1

    _vinyl_rank = sql_vinyl_format_rank("f.formats_json", "f.format_desc")
    vinyl_sql = "" if args.all_formats else f" AND ({_vinyl_rank}) > 0 "
    nfs = "COALESCE(m.num_for_sale, m.release_num_for_sale, 0)"
    floor = "COALESCE(m.release_lowest_price, 0)"
    year_order = "f.year ASC NULLS LAST"
    rid_order = "f.release_id ASC"
    vintage_key = ""
    if vintage_y > 0:
        # 0 = year missing or < cutoff (older / unknown catalog) first; 1 = newer.
        vintage_key = (
            f"(CASE WHEN f.year IS NULL OR f.year < {vintage_y} THEN 0 ELSE 1 END) ASC, "
        )
    if args.sort_by == "listing_floor":
        core_order = f"{floor} DESC, {nfs} DESC, {year_order}, {rid_order}"
    else:
        core_order = f"{nfs} DESC, {floor} DESC, {year_order}, {rid_order}"
    order_clause = f"{vintage_key}{core_order}"

    sql = f"""
        SELECT f.release_id AS release_id
        FROM releases_features f
        INNER JOIN mdb.marketplace_stats m ON m.release_id = f.release_id
        WHERE {nfs} >= ?{vinyl_sql}
        ORDER BY {order_clause}
    """
    lim_sql = ""
    params: list[object] = [int(args.min_num_for_sale)]
    if args.limit and args.limit > 0:
        lim_sql = " LIMIT ?"
        params.append(int(args.limit))

    rows: list[sqlite3.Row] = []
    interrupt_requested = False

    def _on_sigint(_signum: int, _frame: object | None) -> None:
        nonlocal interrupt_requested
        interrupt_requested = True

    def _progress_abort() -> int:
        return 1 if interrupt_requested else 0

    prev_sig = signal.signal(signal.SIGINT, _on_sigint)
    try:
        with _connect(fs) as conn_fs:
            # Lets Ctrl+C abort during long native query execution.
            conn_fs.set_progress_handler(_progress_abort, 20000)
            try:
                conn_fs.execute("ATTACH DATABASE ? AS mdb", (str(mp.resolve()),))
                cur = conn_fs.execute(sql + lim_sql, params)
                rows = cur.fetchall()
            except sqlite3.OperationalError as exc:
                if interrupt_requested or "interrupt" in str(exc).lower():
                    print(
                        "Interrupted; query cancelled (output file not written).",
                        file=sys.stderr,
                    )
                    return 130
                raise
            finally:
                conn_fs.set_progress_handler(None, 0)
    finally:
        signal.signal(signal.SIGINT, prev_sig)

    if not rows:
        print(
            "SQL returned 0 rows (check --min-num-for-sale, vinyl filter unless "
            "--all-formats, and marketplace_stats join).",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(str(r["release_id"]) + "\n")

    scope = "all formats" if args.all_formats else "vinyl only"
    vintage_note = (
        f", vintage_before={vintage_y}" if vintage_y > 0 else ", vintage_first=off"
    )
    print(
        f"Wrote {len(rows)} release_id(s) → {args.out} "
        f"(min_num_for_sale={args.min_num_for_sale}, {scope}, sort_by={args.sort_by}"
        f"{vintage_note})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
