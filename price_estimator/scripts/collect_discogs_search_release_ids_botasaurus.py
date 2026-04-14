#!/usr/bin/env python3
"""
Collect Discogs **website** search release IDs (Vinyl + decade facet).

For each decade, every calendar year in that decade (through the current year,
so the 2020s stop at 2026 instead of 2029) is combined with each HTML sort mode
(relevance, have, want, trending). Up to ``--max-pages`` pages (default **3**)
are fetched per ``(year, sort)``. URLs look like::

    …&format_exact=Vinyl&decade={D}&year={Y}&sort=have,desc&page={p}

IDs are parsed from ``href="/release/{id}-…"`` (see ``discogs_search_release_ids``).
Writes **one newline-delimited file per decade** under ``price_estimator/data/raw/``
(e.g. ``discogs_search_vinyl_decade_2020_release_ids.txt``), deduped at the end
of each decade run.

**Terms of use:** Automated access to the Discogs website may be prohibited or
restricted. Review current policies and rate limits before large runs.

Navigation and Cloudflare handling mirror
``collect_sale_history_botasaurus.py`` (minimal duplicate of that module's
``driver.get`` / bypass helpers). Release IDs are read from the live DOM via
**CDP ``DOM.getOuterHTML``** (``driver.select(…).html``), not only ``run_js``:
Chrome/Botasaurus often omit or truncate structured return values from
``runtime.evaluate``, which produced empty ID lists even when the visible page
had results.

Example::

    PYTHONPATH=. python price_estimator/scripts/collect_discogs_search_release_ids_botasaurus.py \\
        --out-dir price_estimator/data/raw --delay 1.2 --no-headless

    # If Cloudflare blocks headless/automated bypass, solve the challenge in the
    # visible window and confirm in the terminal:
    PYTHONPATH=. python price_estimator/scripts/collect_discogs_search_release_ids_botasaurus.py \\
        --no-headless --manual-cloudflare --decades 2020 --max-pages 3

    PYTHONPATH=. python price_estimator/scripts/collect_discogs_search_release_ids_botasaurus.py \\
        --profile ~/chrome-profiles/discogs --max-per-decade 5000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_DEFAULT_DECADES = (2020, 2010, 2000, 1990, 1980, 1970, 1960)


class _Out:
    verbose: bool = False


out = _Out()


def _pe_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _pe_root().parent


def _ensure_repo_path() -> None:
    r = _repo_root()
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))


def _resolve_out_dir(p: Path | None) -> Path:
    pe = _pe_root()
    if p is None:
        return pe / "data" / "raw"
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "price_estimator":
        return _repo_root() / p
    return pe / p


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _detail(msg: str) -> None:
    if out.verbose:
        print(msg, file=sys.stderr, flush=True)


def _line_buffer_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass


_JS_CF_CHALLENGE_KIND = r"""
(function () {
  const t = (document.title || "").trim().toLowerCase();
  if (t.includes("just a moment")) return "interstitial_title";
  if (t.includes("attention required") || t.includes("access denied"))
    return "title_block";
  const body = (document.body && document.body.innerText) || "";
  const blow = body.toLowerCase();
  if (blow.includes("checking your browser") && blow.includes("cloudflare"))
    return "checking_browser";
  const h =
    (document.documentElement && document.documentElement.innerHTML) || "";
  const hlow = h.toLowerCase();
  if (hlow.includes("why have i been blocked")) return "blocked_page";
  if (hlow.includes("cf-challenge") && hlow.includes("cloudflare")) return "challenge_markup";
  if (hlow.includes("challenges.cloudflare.com")) return "challenge_iframe";
  return "none";
})()
"""


def _cloudflare_challenge_kind(driver) -> str:
    try:
        v = driver.run_js(_JS_CF_CHALLENGE_KIND)
    except Exception:
        return "unknown_js_error"
    if not isinstance(v, str):
        return "none"
    return v.strip() or "none"


def _search_cf_blocks(kind: str) -> bool:
    if kind in ("none", "unknown_js_error"):
        return False
    return kind in (
        "interstitial_title",
        "checking_browser",
        "title_block",
        "blocked_page",
        "challenge_markup",
        "challenge_iframe",
    )


def _cf_banner(*, kind: str, url: str, hint: str) -> None:
    print(
        "\n*** Cloudflare / bot challenge detected ***\n"
        f"  kind: {kind}\n"
        f"  url: {url}\n"
        f"  {hint}\n",
        file=sys.stderr,
        flush=True,
    )


def _is_cloudflare_just_a_moment_title(driver) -> bool:
    try:
        return (driver.title or "").strip() == "Just a moment..."
    except Exception:
        return False


def _discogs_get(
    driver,
    url: str,
    *,
    timeout: float,
    force_bypass_inside_get: bool,
    skip_detect_bypass: bool,
) -> None:
    if force_bypass_inside_get:
        driver.get(url, bypass_cloudflare=True, timeout=timeout)
        return
    driver.get(url, bypass_cloudflare=False, timeout=timeout)
    if skip_detect_bypass:
        return
    if _is_cloudflare_just_a_moment_title(driver):
        _detail("Cloudflare interstitial — detect_and_bypass_cloudflare once …")
        driver.detect_and_bypass_cloudflare()


def _discogs_get_resilient(
    driver,
    url: str,
    *,
    timeout: float,
    force_bypass_inside_get: bool,
    skip_detect_bypass: bool,
) -> None:
    for attempt in (1, 2):
        try:
            _discogs_get(
                driver,
                url,
                timeout=timeout,
                force_bypass_inside_get=force_bypass_inside_get,
                skip_detect_bypass=skip_detect_bypass,
            )
            return
        except BaseException as e:
            msg = str(e).lower()
            if attempt == 1 and any(
                s in msg
                for s in (
                    "no longer available",
                    "target closed",
                    "disconnected",
                    "connection",
                )
            ):
                time.sleep(2.0)
                continue
            raise


def _parse_decades(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid decade token: {p!r}")
        out.append(int(p))
    return tuple(out)


def _load_ids_from_file(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    seen: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.strip()
            if t.isdigit():
                seen.add(t)
    return seen


def _dedupe_output_file_inplace(path: Path) -> None:
    """Rewrite ``path`` as sorted unique digit lines (stable dedupe after a decade)."""
    if not path.is_file():
        return
    ids = _load_ids_from_file(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(ids, key=int):
            f.write(rid + "\n")


def _run_js_ids(driver, js_src: str) -> list[str]:
    raw = driver.run_js(js_src)
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, str) and x.isdigit():
            out.append(x)
        elif isinstance(x, int):
            out.append(str(int(x)))
    return out


# Prefer smaller subtrees first; ``body`` last (cf. sale-history CDP notes).
_CDP_HTML_SELECTORS = (
    "#search_results",
    '[data-testid="search-results"]',
    "#__next",
    '[role="main"]',
    "main",
    "body",
)


def _extract_release_ids_via_cdp(driver) -> list[str]:
    """
    Parse release IDs from outerHTML fetched over CDP.

    Mirrors ``collect_sale_history_botasaurus._pull_sale_history_html``:
    ``run_js`` return values are unreliable for DOM-sized payloads; ``el.html``
    does not depend on JSON serialization of the evaluated expression.
    """
    fn = getattr(_extract_release_ids_via_cdp, "_extract", None)
    if fn is None:
        from price_estimator.src.scrape.discogs_search_release_ids import (
            extract_release_ids_from_html,
        )

        setattr(_extract_release_ids_via_cdp, "_extract", extract_release_ids_from_html)
        fn = extract_release_ids_from_html

    for sel in _CDP_HTML_SELECTORS:
        try:
            el = driver.select(sel, wait=0)
        except Exception:
            el = None
        if el is None:
            continue
        try:
            raw = el.html
        except Exception:
            continue
        if not isinstance(raw, str) or len(raw) < 80:
            continue
        try:
            ids = fn(raw, scope_selector=None)
        except Exception:
            continue
        if not ids:
            continue
        # ``body`` / ``#__next`` can briefly expose only header/footer ``/release/`` links.
        shallow = sel in ("body", "main", "#__next", '[role="main"]')
        if shallow and len(ids) < 8:
            continue
        return ids
    return []


def _scrape_one_search_url(
    driver,
    url: str,
    *,
    label: str,
    js_src: str,
    manual_cloudflare: bool,
    bypass_cloudflare: bool,
    on_cloudflare_stop: bool,
    http_timeout: float,
    backoff: float,
    results_wait: float,
) -> tuple[list[str], int]:
    """
    Navigate, handle CF, poll IDs. Returns ``(ids, exit_code)`` where exit_code
    is 0 (ok), 2 (navigation/poll error), or 3 (CF abort / stop).
    """

    def navigate_once(*, bypass: bool) -> None:
        _discogs_get_resilient(
            driver,
            url,
            timeout=http_timeout,
            force_bypass_inside_get=bypass,
            skip_detect_bypass=manual_cloudflare,
        )

    try:
        navigate_once(bypass=bypass_cloudflare)
    except BaseException as e:
        _progress(f"Navigation error {label}: {e!r}")
        return [], 2

    time.sleep(0.35)
    if manual_cloudflare:
        while True:
            kind0 = _cloudflare_challenge_kind(driver)
            if not _search_cf_blocks(kind0):
                break
            _cf_banner(
                kind=kind0,
                url=url,
                hint="Solve the challenge in the browser, then confirm below.",
            )
            _progress(
                "Complete the Cloudflare challenge in the browser, "
                "then press Enter here to continue scraping."
            )
            try:
                input()
            except EOFError:
                _progress("EOF on stdin; exiting.")
                return [], 3
    else:
        kind0 = _cloudflare_challenge_kind(driver)
        if _search_cf_blocks(kind0):
            _cf_banner(
                kind=kind0,
                url=url,
                hint="Try --no-headless, --profile, or complete challenge.",
            )
            if on_cloudflare_stop:
                return [], 3
            time.sleep(backoff)
            try:
                navigate_once(bypass=True)
            except BaseException as e2:
                _progress(f"CF retry failed: {e2!r}")
                return [], 3
            time.sleep(0.35)

    try:
        page_ids = _poll_release_ids(
            driver,
            js_src,
            max_wait_s=results_wait,
            poll_s=0.45,
        )
    except BaseException as e:
        _progress(f"ID poll failed {label}: {e!r}")
        return [], 2

    if not page_ids:
        _progress(
            f"  {label}: 0 release IDs after {results_wait:.0f}s "
            f"(DOM/CF/cookie banner? try --no-headless --verbose)"
        )
        if out.verbose:
            try:
                t = (driver.title or "").strip()
            except Exception:
                t = "?"
            try:
                u = getattr(driver, "current_url", None) or getattr(driver, "url", None)
            except Exception:
                u = None
            _detail(f"  debug: title={t!r} url={u!r}")

    return page_ids, 0


def _poll_release_ids(
    driver, js_src: str, *, max_wait_s: float, poll_s: float
) -> list[str]:
    """Wait for client-rendered search cards via CDP HTML, then ``run_js`` fallback."""
    deadline = time.monotonic() + max(0.5, float(max_wait_s))
    poll = max(0.15, float(poll_s))
    last: list[str] = []
    while time.monotonic() < deadline:
        try:
            last = _extract_release_ids_via_cdp(driver)
        except BaseException:
            last = []
        if last:
            return last
        try:
            last = _run_js_ids(driver, js_src)
        except BaseException:
            last = []
        if last:
            return last
        time.sleep(poll)
    return last


def main() -> int:
    _ensure_repo_path()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: price_estimator/data/raw)",
    )
    p.add_argument(
        "--decades",
        type=str,
        default=",".join(str(d) for d in _DEFAULT_DECADES),
        help="Comma-separated decade values (default: 2020,...,1960)",
    )
    p.add_argument(
        "--max-per-decade",
        type=int,
        default=20_000,
        help="Stop each decade after this many distinct IDs (default 20000)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Max pages per (calendar year × sort) combo (default 3)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between page navigations (default 1.0)",
    )
    p.add_argument(
        "--http-timeout",
        type=float,
        default=75.0,
        help="Per-navigation timeout seconds (default 75)",
    )
    p.add_argument(
        "--results-wait",
        type=float,
        default=25.0,
        metavar="SEC",
        help=(
            "After each navigation, poll up to SEC for release links in the DOM "
            "(client-rendered search; default 25)"
        ),
    )
    p.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional Chrome user-data dir (persistent profile)",
    )
    p.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window",
    )
    p.add_argument(
        "--bypass-cloudflare",
        action="store_true",
        help="Pass bypass_cloudflare=True on every get (may hang on normal Discogs)",
    )
    p.add_argument(
        "--manual-cloudflare",
        action="store_true",
        help=(
            "Do not run detect_and_bypass_cloudflare after load; when a challenge page "
            "is detected, wait for you to complete it in the browser and press Enter "
            "in this terminal (use with --no-headless or a logged-in --profile)"
        ),
    )
    p.add_argument(
        "--on-cloudflare",
        choices=("stop", "backoff"),
        default="backoff",
        help="When CF challenge detected: stop (exit 3) or backoff once (default)",
    )
    p.add_argument(
        "--cloudflare-backoff-seconds",
        type=float,
        default=90.0,
        help="Sleep before one retry navigation on CF (default 90)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append only new IDs; preload existing lines per output file",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose stderr")
    args = p.parse_args()
    out.verbose = bool(args.verbose)
    _line_buffer_stdio()

    try:
        from shared.project_env import load_project_dotenv
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1
    load_project_dotenv()

    try:
        decades = _parse_decades(args.decades)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = args.profile
    if profile is None:
        raw = (os.environ.get("DISCOGS_SEARCH_BROWSER_PROFILE") or "").strip()
        profile = Path(raw) if raw else None
    if profile is not None:
        profile = profile.expanduser()
        if not profile.is_dir():
            print(f"--profile not a directory: {profile}", file=sys.stderr)
            return 1

    from botasaurus.browser import Driver
    from botasaurus_driver.user_agent import UserAgent
    from botasaurus_driver.window_size import WindowSize

    from price_estimator.src.scrape.discogs_search_release_ids import (
        SEARCH_HTML_SORT_MODES,
        build_vinyl_decade_year_sort_search_url,
        iter_years_for_decade,
        js_collect_release_ids,
    )

    js_src = js_collect_release_ids(scope_selector="#search_results")

    kw: dict = {
        "headless": not args.no_headless,
        "user_agent": UserAgent.HASHED,
        "window_size": WindowSize.HASHED,
        "wait_for_complete_page_load": False,
    }
    if profile is not None:
        kw["profile"] = str(profile)
    driver = Driver(**kw)

    max_d = max(1, int(args.max_per_decade))
    max_pages = max(1, int(args.max_pages))
    delay = max(0.0, float(args.delay))
    http_timeout = max(10.0, float(args.http_timeout))
    backoff = max(1.0, float(args.cloudflare_backoff_seconds))
    results_wait = max(1.0, float(args.results_wait))

    exit_code = 0
    try:
        for decade in decades:
            path = out_dir / f"discogs_search_vinyl_decade_{decade}_release_ids.txt"
            seen = _load_ids_from_file(path) if args.resume else set()
            mode = "a" if args.resume and path.is_file() else "w"
            _progress(
                f"Decade {decade} → {path.name} "
                f"(have={len(seen)}, target<={max_d}, resume={args.resume})"
            )
            years = iter_years_for_decade(decade)
            if years:
                _progress(
                    f"  years {years[0]}..{years[-1]} ({len(years)} yrs) × "
                    f"{len(SEARCH_HTML_SORT_MODES)} sorts × up to {max_pages} pages"
                )
            else:
                _progress(
                    f"  no calendar years in range for decade {decade} "
                    f"(current year caps past decades; skipping inner loops)"
                )
            with open(path, mode, encoding="utf-8", newline="\n") as fp:
                for year in years:
                    if exit_code != 0 or len(seen) >= max_d:
                        break
                    for sort_mode in SEARCH_HTML_SORT_MODES:
                        if exit_code != 0 or len(seen) >= max_d:
                            break
                        for page in range(1, max_pages + 1):
                            if exit_code != 0 or len(seen) >= max_d:
                                break
                            url = build_vinyl_decade_year_sort_search_url(
                                decade=decade,
                                year=year,
                                page=page,
                                sort_mode=sort_mode,
                            )
                            label = (
                                f"decade={decade} year={year} "
                                f"sort={sort_mode} page={page}/{max_pages}"
                            )
                            _detail(url)
                            page_ids, exit_code = _scrape_one_search_url(
                                driver,
                                url,
                                label=label,
                                js_src=js_src,
                                manual_cloudflare=bool(args.manual_cloudflare),
                                bypass_cloudflare=bool(args.bypass_cloudflare),
                                on_cloudflare_stop=args.on_cloudflare == "stop",
                                http_timeout=http_timeout,
                                backoff=backoff,
                                results_wait=results_wait,
                            )
                            if exit_code != 0:
                                break

                            added = 0
                            for rid in page_ids:
                                if len(seen) >= max_d:
                                    break
                                if rid in seen:
                                    continue
                                seen.add(rid)
                                fp.write(rid + "\n")
                                added += 1
                            fp.flush()

                            _detail(f"  {label}: +{added} new (total {len(seen)})")
                            if not page_ids:
                                break
                            if delay > 0:
                                time.sleep(delay)
            _dedupe_output_file_inplace(path)
            _progress(
                f"  deduped → {path.name} ({len(_load_ids_from_file(path))} unique ids)"
            )
            if exit_code != 0:
                break
    finally:
        try:
            driver.close()
        except Exception:
            pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
