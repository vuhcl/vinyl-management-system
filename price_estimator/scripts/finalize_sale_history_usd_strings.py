#!/usr/bin/env python3
r"""
One-shot migration for ``release_sale`` in ``sale_history.sqlite``.

Steps:

1. Infer **USD per EUR** (median, banded) from USD listing + EUR user column.
2. Add/fill temporary ``price_user_usd_approx`` (EUR×rate or parsed user USD).
3. Rewrite ``price_user_currency_text`` to ``$xx.xx``:

   - ``price_original_text`` contains **£** → ``gbp * --gbp-usd`` (default 1.3515).
   - Else → ``price_user_usd_approx``; if null, parse EUR/USD from user text.

4. **DROP** ``price_user_usd_approx``.

Backup the DB first.

Examples::

    PYTHONPATH=. python price_estimator/scripts/finalize_sale_history_usd_strings.py \\
        --db price_estimator/data/cache/sale_history.sqlite --dry-run

    PYTHONPATH=. python price_estimator/scripts/finalize_sale_history_usd_strings.py \\
        --db price_estimator/data/cache/sale_history.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _price_estimator_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _price_estimator_root().parent


def _ensure_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _resolve_db_path(arg: Path | None) -> Path:
    pe = _price_estimator_root()
    if arg is None:
        return pe / "data" / "cache" / "sale_history.sqlite"
    p = Path(arg)
    if p.is_absolute():
        return p
    repo = _repo_root()
    if p.parts and p.parts[0] == "price_estimator":
        return repo / p
    return pe / p


def _sqlite_supports_drop_column(conn: sqlite3.Connection) -> bool:
    ver = conn.execute("SELECT sqlite_version()").fetchone()[0]
    parts = str(ver).split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return False
    return major > 3 or (major == 3 and minor >= 35)


def main() -> int:
    _ensure_path()
    from price_estimator.src.scrape.sale_history_currency import (
        collect_usd_eur_ratios_from_rows,
        format_usd_money_string,
        parse_eur_amount,
        parse_gbp_amount,
        parse_usd_user_amount,
        usd_per_eur_from_pairs,
    )
    from price_estimator.src.storage.sale_history_db import SaleHistoryDB

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to sale_history.sqlite",
    )
    p.add_argument(
        "--gbp-usd",
        type=float,
        default=1.3515,
        help="GBP listing × this → USD user string (default 1.3515)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Calibration + counts only",
    )
    p.add_argument(
        "--usd-per-eur",
        type=float,
        default=None,
        help="Override inferred USD-per-EUR",
    )
    p.add_argument("--ratio-lo", type=float, default=0.95)
    p.add_argument("--ratio-hi", type=float, default=1.35)
    args = p.parse_args()

    db_path = _resolve_db_path(args.db)
    SaleHistoryDB(db_path)

    gbp_usd = float(args.gbp_usd)

    conn = sqlite3.connect(str(db_path), timeout=120.0)
    conn.row_factory = sqlite3.Row

    cols = {r[1] for r in conn.execute("PRAGMA table_info(release_sale)")}
    if "price_user_usd_approx" not in cols:
        conn.execute(
            "ALTER TABLE release_sale ADD COLUMN price_user_usd_approx REAL"
        )
        conn.commit()

    t0 = time.perf_counter()
    cur = conn.execute(
        """
        SELECT price_original_text, price_user_currency_text
        FROM release_sale
        WHERE price_user_currency_text LIKE '%€%'
          AND price_original_text LIKE '%$%'
          AND price_original_text NOT LIKE 'A$%'
          AND price_original_text NOT LIKE 'CA$%'
          AND price_original_text NOT LIKE 'C$%'
          AND price_original_text NOT LIKE 'AU$%'
        """
    )
    ratios = collect_usd_eur_ratios_from_rows(
        ((r[0] or "", r[1] or "") for r in cur)
    )
    dt = time.perf_counter() - t0
    print(f"Calibration scan {dt:.1f}s; n_ratios={len(ratios)}", flush=True)

    used: list[float] = []
    if args.usd_per_eur is not None:
        usd_per_eur = float(args.usd_per_eur)
        print(f"Using override usd_per_eur={usd_per_eur:.6f}", flush=True)
    else:
        inferred, used = usd_per_eur_from_pairs(
            ratios, lo=args.ratio_lo, hi=args.ratio_hi
        )
        if not used or inferred is None:
            msg = (
                "No EUR/USD ratios in band; pass --usd-per-eur or "
                "widen --ratio-lo/--ratio-hi"
            )
            print(msg, file=sys.stderr)
            conn.close()
            return 1
        usd_per_eur = inferred
        print(
            f"Inferred usd_per_eur={usd_per_eur:.6f} from n={len(used)} pairs",
            flush=True,
        )

    cur_gbp = conn.execute(
        "SELECT COUNT(*) FROM release_sale WHERE price_original_text LIKE '%£%'"
    )
    n_gbp = cur_gbp.fetchone()[0]
    print(
        f"Rows with £ in original: {n_gbp}; gbp_usd multiplier={gbp_usd}",
        flush=True,
    )

    if args.dry_run:
        print("Dry run: no UPDATE/DROP.", flush=True)
        conn.close()
        return 0

    def sh_user_to_usd(text: str | None, rate: float | None) -> float | None:
        if rate is None or rate <= 0:
            return None
        t = text or ""
        e = parse_eur_amount(t)
        if e is not None and e > 0:
            return float(e) * float(rate)
        u = parse_usd_user_amount(t)
        if u is not None and u > 0:
            return float(u)
        return None

    conn.create_function("sh_user_to_usd", 2, sh_user_to_usd)

    def sh_format_user_usd(
        orig: str | None,
        approx: float | None,
        user_txt: str | None,
    ) -> str | None:
        o = orig if isinstance(orig, str) else (orig or "")
        u = user_txt if isinstance(user_txt, str) else (user_txt or "")
        g = parse_gbp_amount(o)
        if g is not None and g > 0:
            try:
                return format_usd_money_string(float(g) * gbp_usd)
            except ValueError:
                return None
        try:
            af = float(approx) if approx is not None else float("nan")
        except (TypeError, ValueError):
            af = float("nan")
        if af == af and af > 0:
            try:
                return format_usd_money_string(af)
            except ValueError:
                return None
        e = parse_eur_amount(u)
        if e is not None and e > 0 and usd_per_eur and usd_per_eur > 0:
            try:
                return format_usd_money_string(float(e) * float(usd_per_eur))
            except ValueError:
                return None
        uu = parse_usd_user_amount(u)
        if uu is not None and uu > 0:
            try:
                return format_usd_money_string(float(uu))
            except ValueError:
                return None
        return None

    conn.create_function("sh_format_user_usd", 3, sh_format_user_usd)

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        """
        UPDATE release_sale
        SET price_user_usd_approx = sh_user_to_usd(price_user_currency_text, ?)
        """,
        (usd_per_eur,),
    )
    n_approx = int(conn.execute("SELECT changes()").fetchone()[0])

    conn.execute(
        """
        UPDATE release_sale
        SET price_user_currency_text = COALESCE(
            sh_format_user_usd(
                price_original_text,
                price_user_usd_approx,
                price_user_currency_text
            ),
            price_user_currency_text
        )
        """
    )
    n_txt = int(conn.execute("SELECT changes()").fetchone()[0])

    if not _sqlite_supports_drop_column(conn):
        conn.rollback()
        conn.close()
        msg = (
            "SQLite < 3.35: cannot DROP COLUMN; upgrade SQLite or rebuild "
            "release_sale without price_user_usd_approx."
        )
        print(msg, file=sys.stderr)
        return 1

    conn.execute("ALTER TABLE release_sale DROP COLUMN price_user_usd_approx")

    now = datetime.now(timezone.utc).isoformat()
    p05_v = p50_v = p95_v = None
    if len(used) >= 20:
        qs = statistics.quantiles(used, n=20)
        med_u = float(statistics.median(used))
        p05_v, p50_v, p95_v = float(qs[0]), med_u, float(qs[18])
    elif used:
        p50_v = float(statistics.median(used))
    conn.execute(
        """
        INSERT INTO sale_history_calibration (
            computed_at, usd_per_eur, n_pairs, ratio_p05, ratio_p50, ratio_p95
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (now, usd_per_eur, len(used) if used else 0, p05_v, p50_v, p95_v),
    )
    conn.commit()
    conn.close()

    print(
        f"Done. filled_approx~{n_approx}; updated_text~{n_txt}; "
        f"dropped column (gbp_usd={gbp_usd}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
