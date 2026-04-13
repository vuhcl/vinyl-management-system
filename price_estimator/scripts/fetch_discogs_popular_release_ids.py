#!/usr/bin/env python3
"""
Fetch Discogs release IDs sorted by community **have** (most-owned), matching the
website intent of::

  /search?type=release&sort=have,desc

**Recommended:** use the authenticated **REST API** (same ordering as the site for
this sort). ``per_page`` is capped at **100** per request (the site’s
``limit=250`` is HTML-only).

  uv run python price_estimator/scripts/fetch_discogs_popular_release_ids.py \\
      --out price_estimator/data/raw/popular_by_have_page1.txt --max-pages 1

  # All pages (can be slow / rate-limited)
  uv run python price_estimator/scripts/fetch_discogs_popular_release_ids.py \\
      --out price_estimator/data/raw/popular_by_have_all.txt

**Optional HTML:** pass ``--html-url`` with a full search URL (e.g. the one you
use in the browser). This uses ``requests`` + regex on ``href="/release/<id>``;
it can break if Discogs changes markup and may rate-limit more aggressively than
the API.

Env: ``DISCOGS_USER_TOKEN`` or ``DISCOGS_TOKEN`` (API mode only).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import requests


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_ids_from_api(
    *,
    max_pages: int,
    per_page: int,
    token: str,
) -> Iterable[str]:
    from shared.discogs_api.client import DiscogsClient

    client = DiscogsClient(user_token=token)
    page = 1
    while True:
        if max_pages > 0 and page > max_pages:
            break
        data = client.database_search(
            query="",
            result_type="release",
            page=page,
            per_page=per_page,
            sort="have",
            sort_order="desc",
        )
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            break
        for row in results:
            if not isinstance(row, dict):
                continue
            if row.get("type") != "release":
                continue
            rid = row.get("id")
            if rid is not None:
                yield str(int(rid))

        pag = (data or {}).get("pagination") or {}
        pages = int(pag.get("pages") or 1)
        if page >= pages:
            break
        page += 1


def iter_ids_from_html(url: str) -> Iterable[str]:
    """
    Best-effort scrape: result cards use ``href="/release/<id>-<slug>"``.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; VinylManagementSystem/1.0; +local script)"
        ),
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    html = r.text
    seen: set[str] = set()
    # Typical: href="/release/4570366-Random-Access-Memories"
    for m in re.finditer(r'href="/release/(\d+)-', html):
        rid = m.group(1)
        if rid not in seen:
            seen.add(rid)
            yield rid
    if not seen:
        for m in re.finditer(r'href="/release/(\d+)"', html):
            rid = m.group(1)
            if rid not in seen:
                seen.add(rid)
                yield rid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Discogs release IDs (popular by 'have', descending).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write one release_id per line (default: stdout)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="API mode: max search pages (0 = fetch all pages)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="API mode: per_page (1–100)",
    )
    parser.add_argument(
        "--html-url",
        type=str,
        default=None,
        help="If set, scrape this URL instead of using the API",
    )
    args = parser.parse_args()

    if args.html_url:
        try:
            ids = list(iter_ids_from_html(args.html_url))
        except requests.RequestException as e:
            print(f"HTML fetch failed: {e}", file=sys.stderr)
            return 1
    else:
        try:
            from shared.project_env import load_project_dotenv

            load_project_dotenv()
        except ImportError:
            print("PYTHONPATH must include repo root.", file=sys.stderr)
            return 1
        from shared.discogs_api.client import personal_access_token_from_env

        token = personal_access_token_from_env()
        if not token:
            print(
                "Set DISCOGS_USER_TOKEN or DISCOGS_TOKEN for API mode, "
                "or use --html-url.",
                file=sys.stderr,
            )
            return 1
        pp = max(1, min(int(args.per_page), 100))
        ids = list(
            iter_ids_from_api(
                max_pages=args.max_pages,
                per_page=pp,
                token=token,
            )
        )

    if args.out:
        args.out = args.out if args.out.is_absolute() else _repo_root() / args.out
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for rid in ids:
                f.write(rid + "\n")
        print(f"Wrote {len(ids)} release_id lines → {args.out}", file=sys.stderr)
    else:
        for rid in ids:
            print(rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
