#!/usr/bin/env python3
"""
Collect Discogs **release marketplace** listings from the website for grader data.

Target URL pattern::

    https://www.discogs.com/sell/release/{release_id}?sort=price%2Casc&limit=250&page={n}

**Chrome profile (required):** Same pattern as ``collect_sale_history_botasaurus.py``:
persistent user-data dir with Discogs login, ``--no-headless`` for first-time sign-in,
optional ``DISCOGS_SALE_HISTORY_BROWSER_PROFILE`` / ``--profile``.
**Cloudflare:** navigate with ``bypass_cloudflare=False``; on ``Just a moment...`` title,
``detect_and_bypass_cloudflare()`` once after ``get`` (see sale-history collector).

**Terms of use:** Automated access to the Discogs website may be prohibited or
restricted. Review Discogs' current terms and developer policies and obtain
appropriate permission before running this collector at scale.

**``--out-dir`` paths:** Relative paths are resolved from the **monorepo root**
(parent of ``price_estimator/``).

**Resume:** Skips re-fetching when ``{out_dir}/{release_id}/page_{n:03d}.html`` exists
and is non-empty (use ``--no-resume`` to force re-download).

**Pacing:** ``--delay`` / ``--between-releases-delay`` (seconds after each release before
the next); ``--between-pages-delay`` overrides the pause between pagination requests
for the same release (default ``max(0.3, 0.25 * delay)``).

**Hand-picked queue:** Merge ``release_scrape_queue_auto.txt`` + ``release_scrape_queue_manual.txt``
with ordered dedupe (see project plan), then pass the final file as ``--release-ids``.

**Progress:** By default one checkpoint line every ``--progress-every`` releases (200);
use ``0`` for legacy per-page lines. DOM wait detail only with ``--verbose`` unless
legacy mode.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_release_ids_from_file(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            token = s.split()[0]
            if token.isdigit():
                yield token


def ordered_unique_release_ids(path: Path) -> list[str]:
    return list(dict.fromkeys(iter_release_ids_from_file(path)))


def _sale_history_shim():
    """Load navigation helpers from the sale-history script (no package cycle)."""
    p = Path(__file__).resolve().parent / "collect_sale_history_botasaurus.py"
    spec = importlib.util.spec_from_file_location("_discogs_sale_history_bot_mod", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _wrap_page_html(inner: str) -> str:
    return "<!DOCTYPE html><html><body>" + inner + "</body></html>"


def _pull_release_marketplace_html(
    driver,
    *,
    _detail,
    max_wait: float = 40.0,
    poll: float = 0.45,
) -> str:
    """Wait for marketplace DOM (#pjax_container or item rows), then CDP outerHTML."""
    deadline = time.monotonic() + float(max_wait)
    last_log = 0.0
    polls = 0
    time.sleep(0.35)

    def try_extract() -> str | None:
        for selector, label in (
            ("#pjax_container", "#pjax_container"),
            ("#page_content", "#page_content"),
        ):
            try:
                el = driver.select(selector, wait=0)
            except Exception:
                el = None
            if el is None:
                continue
            try:
                raw = el.html
            except Exception:
                continue
            if not isinstance(raw, str) or len(raw) <= 400:
                continue
            low = raw.lower()
            if "item_description" in low or "/sell/item/" in low:
                return raw
        return None

    while time.monotonic() < deadline:
        polls += 1
        got = try_extract()
        if got is not None:
            _detail(
                f"         Marketplace DOM via CDP ({len(got) // 1024} KiB) after "
                f"{polls} poll(s); parsing …"
            )
            return _wrap_page_html(got)
        now = time.monotonic()
        if now - last_log >= 6.0:
            _detail(
                f"         Waiting for marketplace table (poll #{polls}, "
                f"every {poll:.0f}s) …"
            )
            last_log = now
        time.sleep(poll)

    try:
        el = driver.wait_for_element("#pjax_container", wait=10)
        raw = el.html
        if isinstance(raw, str) and len(raw) > 400:
            return _wrap_page_html(raw)
    except Exception:
        pass

    return _wrap_page_html("")


def _cloudflare_blocks_release(kind: str, html: str, parsed) -> bool:
    from price_estimator.src.scrape.discogs_sale_history_parse import looks_like_login_or_challenge

    if kind in ("none", "unknown_js_error"):
        return False
    if kind in (
        "interstitial_title",
        "checking_browser",
        "title_block",
        "blocked_page",
    ):
        return True
    warns = getattr(parsed, "parse_warnings", []) or []
    parse_bad = looks_like_login_or_challenge(html) or (
        "no_listing_rows_matched" in warns
    )
    if not parse_bad:
        return False
    return kind in ("challenge_markup", "challenge_iframe")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Discogs /sell/release marketplace pages (Botasaurus + profile)",
    )
    parser.add_argument(
        "--release-ids",
        type=Path,
        required=True,
        help="Text file: one release ID per line (# comments); merged auto+manual list ok",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output root (default: grader/data/raw/discogs/release_marketplace "
            "resolved from monorepo root). Per-release: {out_dir}/{rid}/page_*.html "
            "and listings_raw.ndjson"
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Chrome user-data dir (default: env DISCOGS_SALE_HISTORY_BROWSER_PROFILE)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window (first-time Discogs login)",
    )
    parser.add_argument(
        "--login-wait-seconds",
        type=float,
        default=-1.0,
        metavar="SEC",
        help="Seconds to wait for sign-in on discogs.com before first scrape (-1 = 120 if --no-headless else 0)",
    )
    parser.add_argument(
        "--login-pause",
        action="store_true",
        help="Block until Enter after opening discogs.com (unlimited login time)",
    )
    parser.add_argument(
        "--assume-logged-in",
        action="store_true",
        help="Skip homepage; go straight to marketplace URLs (profile must be signed in)",
    )
    parser.add_argument(
        "--bypass-cloudflare",
        action="store_true",
        help="Pass bypass_cloudflare=True on every get (may hang; default matches sale history)",
    )
    parser.add_argument(
        "--delay",
        "--between-releases-delay",
        type=float,
        default=4.0,
        metavar="SEC",
        dest="delay",
        help="Seconds to wait after each release (all its pages) before the next (default 4)",
    )
    parser.add_argument(
        "--between-pages-delay",
        type=float,
        default=None,
        metavar="SEC",
        dest="between_pages_delay",
        help=(
            "Seconds before fetching the next marketplace page for the same release "
            "(default: max(0.3, 0.25 * --delay))"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=250,
        help="Items per page (query limit=), max 250 (default 250)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="price,asc",
        help="Sort query value (default price,asc)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Max marketplace pages per release (default 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip fetching a page if page_*.html already exists and non-empty",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force re-fetch all pages (overrides --resume)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose stderr (includes marketplace DOM wait detail when not using --progress-every 0)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        metavar="N",
        help=(
            "Print one checkpoint line every N releases completed (default 200). "
            "Also prints after the final release. Use 0 for legacy per-fetch/per-page logs."
        ),
    )
    parser.add_argument(
        "--on-cloudflare",
        choices=("stop", "backoff"),
        default="stop",
        help="Cloudflare handling after navigation+parse (same semantics as sale history)",
    )
    parser.add_argument(
        "--cloudflare-backoff-seconds",
        type=float,
        default=120.0,
        metavar="SEC",
        help="Backoff before retry when --on-cloudflare backoff",
    )
    args = parser.parse_args()
    _page_gap = (
        max(0.0, float(args.between_pages_delay))
        if args.between_pages_delay is not None
        else max(0.3, float(args.delay) * 0.25)
    )
    _release_gap = max(0.0, float(args.delay))

    try:
        from shared.project_env import load_project_dotenv
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    load_project_dotenv()

    progress_every = int(args.progress_every)
    if progress_every < 0:
        print("--progress-every must be >= 0", file=sys.stderr)
        return 1
    legacy_progress = progress_every == 0

    shim = _sale_history_shim()
    shim.out.verbose = bool(args.verbose)
    shim._line_buffer_stdio()
    _progress = shim._progress
    _detail = shim._detail
    _detail_log = _detail if (args.verbose or legacy_progress) else (lambda *_a, **_k: None)
    _discogs_get_resilient = shim._discogs_get_resilient
    _heartbeat_while = shim._heartbeat_while
    _wait_for_discogs_login = shim._wait_for_discogs_login
    _cloudflare_challenge_kind = shim._cloudflare_challenge_kind
    _cloudflare_stderr_banner = shim._cloudflare_stderr_banner

    repo = _repo_root()
    out_root = args.out_dir
    if out_root is None:
        out_root = repo / "grader" / "data" / "raw" / "discogs" / "release_marketplace"
    elif not out_root.is_absolute():
        out_root = (repo / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    profile = args.profile
    if profile is None:
        raw = (os.environ.get("DISCOGS_SALE_HISTORY_BROWSER_PROFILE") or "").strip()
        profile = Path(raw) if raw else None
    if not profile or not str(profile).strip():
        print(
            "Set --profile or DISCOGS_SALE_HISTORY_BROWSER_PROFILE to a Chrome user-data "
            "directory where you are logged into Discogs.",
            file=sys.stderr,
        )
        return 1
    profile = profile.expanduser()
    if not profile.is_dir():
        print(f"Profile directory does not exist: {profile}", file=sys.stderr)
        return 1

    rp = args.release_ids.expanduser()
    if not rp.is_file():
        print(f"--release-ids not found: {rp}", file=sys.stderr)
        return 1

    ids = ordered_unique_release_ids(rp)
    if not ids:
        print(f"No numeric release IDs in {rp}", file=sys.stderr)
        return 1

    resume = bool(args.resume) and not bool(args.no_resume)

    from botasaurus.browser import Driver
    from botasaurus_driver.user_agent import UserAgent
    from botasaurus_driver.window_size import WindowSize

    from price_estimator.src.scrape.discogs_release_listings_parse import (
        parse_release_listings_html,
        release_marketplace_url,
    )
    from price_estimator.src.scrape.discogs_sale_history_parse import looks_like_login_or_challenge

    driver = Driver(
        headless=not args.no_headless,
        profile=str(profile),
        user_agent=UserAgent.HASHED,
        window_size=WindowSize.HASHED,
        wait_for_complete_page_load=False,
    )

    login_wait = args.login_wait_seconds
    if login_wait < 0:
        login_wait = 120.0 if args.no_headless else 0.0

    exit_code = 0
    ok = err = 0

    try:
        if not args.assume_logged_in and (args.login_pause or login_wait > 0):
            _progress("Opening https://www.discogs.com/ …")
            hb_stop = threading.Event()
            hb = threading.Thread(
                target=_heartbeat_while,
                args=("loading discogs.com", hb_stop),
                daemon=True,
            )
            hb.start()
            try:
                _discogs_get_resilient(
                    driver,
                    "https://www.discogs.com/",
                    timeout=75,
                    force_bypass_inside_get=args.bypass_cloudflare,
                )
            finally:
                hb_stop.set()
            if args.login_pause:
                _progress("Press Enter when logged in …")
                try:
                    input()
                except EOFError:
                    pass
            else:
                _wait_for_discogs_login(driver, float(login_wait))

        for i, rid in enumerate(ids, start=1):
            rel_dir = out_root / rid
            rel_dir.mkdir(parents=True, exist_ok=True)
            ndjson_path = rel_dir / "listings_raw.ndjson"
            page = 1
            max_cf = 2 if args.on_cloudflare == "backoff" else 1
            release_aborted = False

            while page <= max(1, args.max_pages):
                url = release_marketplace_url(
                    rid, page=page, limit=args.limit, sort=args.sort
                )
                page_path = rel_dir / f"page_{page:03d}.html"
                skip_ndjson_append = False
                html: str | None = None

                if resume and page_path.is_file() and page_path.stat().st_size > 100:
                    html = page_path.read_text(encoding="utf-8", errors="replace")
                    skip_ndjson_append = True
                    _detail_log(
                        f"[{i}/{len(ids)}] {rid} page {page}: resume (cached {page_path.name}, "
                        "no NDJSON append)"
                    )

                if html is None:
                    if legacy_progress:
                        _progress(f"[{i}/{len(ids)}] release {rid} page {page} → fetch")
                    cf_try = 0
                    navigated = False
                    while cf_try < max_cf and not navigated:
                        cf_try += 1
                        hb_stop = threading.Event()
                        hb = threading.Thread(
                            target=_heartbeat_while,
                            args=(f"GET marketplace {rid} p{page}", hb_stop),
                            daemon=True,
                        )
                        hb.start()
                        try:
                            _discogs_get_resilient(
                                driver,
                                url,
                                timeout=60,
                                force_bypass_inside_get=args.bypass_cloudflare,
                            )
                        finally:
                            hb_stop.set()
                        html = _pull_release_marketplace_html(
                            driver, _detail=_detail_log
                        )
                        page_path.write_text(html, encoding="utf-8")

                        parsed_probe = parse_release_listings_html(html, rid, page=page)
                        kind = _cloudflare_challenge_kind(driver)
                        if _cloudflare_blocks_release(kind, html, parsed_probe):
                            _cloudflare_stderr_banner(
                                kind=kind,
                                release_id=rid,
                                url=url,
                                hint_extra=(
                                    "Stopping (--on-cloudflare stop)."
                                    if args.on_cloudflare == "stop"
                                    else f"Backoff {args.cloudflare_backoff_seconds:.0f}s"
                                ),
                            )
                            if args.on_cloudflare == "stop":
                                exit_code = 3
                                err += 1
                                release_aborted = True
                                break
                            if cf_try < max_cf:
                                time.sleep(max(1.0, args.cloudflare_backoff_seconds))
                                continue
                            err += 1
                            release_aborted = True
                            break

                        if looks_like_login_or_challenge(html):
                            print(
                                f"[err] {rid} p{page}: login/challenge page",
                                file=sys.stderr,
                            )
                            err += 1
                            release_aborted = True
                            break

                        navigated = True

                    if release_aborted or exit_code == 3:
                        break

                assert html is not None
                parsed = parse_release_listings_html(html, rid, page=page)
                n_list = len(parsed.listings)
                if not skip_ndjson_append:
                    with open(ndjson_path, "a", encoding="utf-8") as out_f:
                        for row in parsed.listings:
                            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                if legacy_progress:
                    _progress(
                        f"[ok] release {rid} page {page}: parsed {n_list} listing(s) "
                        f"(ndjson append={'no' if skip_ndjson_append else 'yes'})"
                    )

                if n_list == 0:
                    break
                if n_list < args.limit:
                    break
                page += 1
                if _page_gap > 0:
                    time.sleep(_page_gap)

            if exit_code == 3:
                break
            if not release_aborted:
                ok += 1
            if not legacy_progress and progress_every > 0:
                if i % progress_every == 0 or i == len(ids):
                    _progress(
                        f"[{i}/{len(ids)}] checkpoint ok={ok} err={err} last_release={rid}"
                    )
            if i < len(ids) and _release_gap > 0:
                time.sleep(_release_gap)

    finally:
        try:
            driver.close()
        except Exception:
            pass

    _progress(f"Done releases processed={ok} errors={err} out_dir={out_root}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
