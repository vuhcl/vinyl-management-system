#!/usr/bin/env python3
"""
Collect Discogs **sale history** from the website (not available via API).

Target URL (logged-in only)::

    https://www.discogs.com/sell/history/{release_id}

**Chrome profile (required):** create a persistent directory, run once with
``--no-headless``, sign into Discogs in the opened window, then reuse the
same ``--profile`` path. With ``--no-headless``, the script opens Discogs first
and **polls for sign-in** (default up to **120s**); it continues **as soon as**
logout / My Discogs links appear (override with ``--login-wait-seconds`` or use
``--login-pause`` to wait until you press Enter).
Alternatively set env ``DISCOGS_SALE_HISTORY_BROWSER_PROFILE`` to that directory.
If the homepage load hangs past the timeout, use ``--assume-logged-in`` so the
script skips that hop and opens sale-history URLs directly (profile must already
be signed in). **Cloudflare:** we always call ``driver.get(..., bypass_cloudflare=False)``
so Botasaurus does not run its broad detector on every Discogs page (that path can
loop forever). If the document title is Cloudflare's ``Just a moment...`` interstitial,
we then call ``detect_and_bypass_cloudflare()`` once. Use ``--bypass-cloudflare`` only
if you need the old "bypass inside every get" behavior (may hang on normal Discogs).

**Terms of use:** Automated access to the Discogs website may be prohibited or
restricted. Review Discogs' current terms and developer policies and obtain
appropriate permission before running this collector at scale.

**``--db`` paths:** Same rule as ``collect_marketplace_stats.py``: relative values
are resolved from the **monorepo root** (parent of ``price_estimator/``), not from
inside ``price_estimator/``. Example:
``--db price_estimator/data/cache/sale_history.sqlite`` when invoking from the repo
root.

**Pagination:** v1 scrapes the history table as rendered on the first loaded page
only; extend the driver loop if Discogs exposes additional pages (e.g. “next”).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from price_estimator.src.scrape.discogs_sale_history_parse import looks_like_login_or_challenge


class _Output:
    """Default is compact stderr; set ``verbose=True`` for step-by-step diagnostics."""

    verbose: bool = False


out = _Output()


def _repo_root() -> Path:
    """Monorepo root (parent of the ``price_estimator`` package)."""
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
    """Preserve file order, skip duplicate IDs (avoids repeated navigations)."""
    return list(dict.fromkeys(iter_release_ids_from_file(path)))


def _line_buffer_stdio() -> None:
    """Avoid silent terminals when stdout is fully block-buffered (e.g. some ``uv run`` pipes)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass


def _progress(msg: str) -> None:
    """Always printed (compact status, outcomes, errors)."""
    print(msg, file=sys.stderr, flush=True)


def _detail(msg: str) -> None:
    """Verbose-only (DOM polls, heartbeats, sub-steps)."""
    if out.verbose:
        print(msg, file=sys.stderr, flush=True)


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
    """Small probe; returns ``none`` or a short reason token."""
    try:
        v = driver.run_js(_JS_CF_CHALLENGE_KIND)
    except Exception:
        return "unknown_js_error"
    if not isinstance(v, str):
        return "none"
    return v.strip() or "none"


def _cloudflare_stderr_banner(*, kind: str, release_id: str, url: str, hint_extra: str) -> None:
    print(
        "\n*** Cloudflare / bot challenge detected ***\n"
        f"  kind: {kind}\n"
        f"  release_id: {release_id}\n"
        f"  url: {url}\n"
        f"  {hint_extra}\n",
        file=sys.stderr,
        flush=True,
    )


def _cloudflare_blocks_scrape(kind: str, html: str, parsed) -> bool:
    """Whether we should treat the tab as blocked by Cloudflare (not a normal Discogs error)."""
    if kind in ("none", "unknown_js_error"):
        return False
    if kind in (
        "interstitial_title",
        "checking_browser",
        "title_block",
        "blocked_page",
    ):
        return True
    parse_bad = looks_like_login_or_challenge(html) or (
        "sales_table_not_found" in getattr(parsed, "parse_warnings", [])
    )
    if not parse_bad:
        return False
    return kind in ("challenge_markup", "challenge_iframe")


def _is_cloudflare_just_a_moment_title(driver) -> bool:
    """Match Botasaurus ``solve_full_cf`` gate (``driver.title == "Just a moment..."``)."""
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
) -> None:
    """
    Navigate without in-get bypass (avoids spurious Turnstile matches on Discogs).

    If the tab shows Cloudflare's interstitial title, run Botasaurus bypass once —
    that path uses ``solve_full_cf`` instead of the widget loop that can hang.
    """
    if force_bypass_inside_get:
        driver.get(url, bypass_cloudflare=True, timeout=timeout)
        return
    driver.get(url, bypass_cloudflare=False, timeout=timeout)
    if _is_cloudflare_just_a_moment_title(driver):
        _detail(
            '         Cloudflare interstitial (title "Just a moment...") — '
            "running Botasaurus bypass once …"
        )
        driver.detect_and_bypass_cloudflare()


def _discogs_get_resilient(
    driver,
    url: str,
    *,
    timeout: float,
    force_bypass_inside_get: bool,
) -> None:
    """``driver.get`` can detach the CDP target on some navigations; retry once."""
    for attempt in (1, 2):
        try:
            _discogs_get(
                driver,
                url,
                timeout=timeout,
                force_bypass_inside_get=force_bypass_inside_get,
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
                _detail(
                    f"         Navigation CDP error ({e!r}); sleeping 2s and retrying once …"
                )
                time.sleep(2.0)
                continue
            raise


def _heartbeat_while(
    label: str,
    stop: threading.Event,
    interval: float = 10.0,
    *,
    max_lines: int = 12,
) -> None:
    """Print at most ``max_lines`` updates (~2 min at 10s) then one final note."""
    n = 0
    while not stop.wait(timeout=interval):
        n += 1
        if n <= max_lines:
            _detail(f"         … still busy ({label}), {n * interval:.0f}s elapsed …")
        elif n == max_lines + 1:
            _detail(
                f"         … ({label}) still running (no more heartbeat lines). "
                "If this hangs past a few minutes, try Ctrl+C and re-run with "
                "--assume-logged-in if your profile is already signed in."
            )


# Discogs UI probes (run in page context via ``driver.run_js``).
_JS_LOGIN_STATE = r"""
(function () {
  const title = (
    document.querySelector("title") && document.querySelector("title").textContent
  ) || "";
  if (/just a moment/i.test(title)) return "challenge";
  const html = (document.documentElement && document.documentElement.innerHTML) || "";
  const hlow = html.toLowerCase();
  if (hlow.includes("enable javascript and cookies")) return "challenge";
  const path = (location.pathname || "").toLowerCase();
  if (path === "/login" || path.endsWith("/login")) return "logged_out";

  const links = Array.from(document.querySelectorAll("a[href]"));
  for (const a of links) {
    const hr = (a.getAttribute("href") || "").toLowerCase();
    if (hr.includes("logout") || hr.includes("log_out") || hr.includes("signout") || hr.includes("sign_out"))
      return "logged_in";
    if (hr.includes("/my/") || hr.includes("/myorders")) return "logged_in";
    if (hr.includes("users/logout") || hr.includes("user/logout")) return "logged_in";
  }
  for (const a of document.querySelectorAll("a, button, [role='button']")) {
    const t = (a.textContent || "").trim().toLowerCase();
    if (t === "logout" || t === "log out" || t === "sign out") return "logged_in";
  }
  const body = (document.body && document.body.innerText) || "";
  const blow = body.toLowerCase();
  if (/\blog\s*out\b|\bsign\s*out\b/.test(blow)) return "logged_in";
  if (/\bmy\s+discogs\b/i.test(blow)) return "logged_in_maybe";
  if (/\bcollection\b/i.test(blow) && /\bwantlist\b/i.test(blow)) return "logged_in_maybe";

  const hasLogin = !!document.querySelector('a[href*="/login"]');
  if (hasLogin && /sign\s*in/i.test(body)) return "logged_out";
  return "unknown";
})()
"""

def _wrap_sale_history_html(inner: str) -> str:
    return "<!DOCTYPE html><html><body>" + inner + "</body></html>"


def _wait_for_discogs_login(driver, max_seconds: float) -> None:
    """Poll until logout link / my Discogs appears, or ``max_seconds`` elapses."""
    deadline = time.monotonic() + float(max_seconds)
    announced = False
    last_status = 0.0
    while time.monotonic() < deadline:
        try:
            state = driver.run_js(_JS_LOGIN_STATE)
        except Exception:
            state = "unknown"
        if state in ("logged_in", "logged_in_maybe"):
            _progress("Signed in — continuing.")
            return
        if not announced:
            if out.verbose:
                _detail(
                    f"Waiting for Discogs sign-in (checking every 2s, max {max_seconds:.0f}s; "
                    "continues as soon as you are logged in) …"
                )
            else:
                _progress(
                    f"Waiting for Discogs sign-in (up to {max_seconds:.0f}s) …"
                )
            announced = True
        now = time.monotonic()
        if now - last_status >= 12.0:
            left = max(0.0, deadline - now)
            extra = ""
            if state == "challenge":
                extra = " (Cloudflare challenge — finish in the browser) "
            elif state == "logged_out":
                extra = " (still on sign-in — log in in the browser) "
            _detail(f"         …{extra}~{left:.0f}s left …")
            last_status = now
        time.sleep(2.0)
    _progress(
        "Login poll ended — proceeding (if you are not signed in, scrapes may fail)."
    )


def _pull_sale_history_html(
    driver,
    *,
    max_wait_table: float = 35.0,
    poll: float = 0.45,
) -> str:
    """
    Wait for the sale-history table, then fetch HTML via CDP ``DOM.getOuterHTML``.

    Botasaurus ``run_js`` / ``runtime.evaluate`` returns large strings through JSON;
    Chrome often omits ``remote_object.value`` for big ``outerHTML``, so Python sees
    ``None`` and we used to log "0 KiB". Element ``.html`` uses ``get_outer_html`` instead.
    """
    deadline = time.monotonic() + float(max_wait_table)
    last_log = 0.0
    polls = 0
    time.sleep(0.35)

    def try_extract() -> str | None:
        for selector, label in (
            ("table.sales-history-table", "sales-history-table"),
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
            except Exception as exc:
                _detail(f"         {label}: CDP get_outer_html failed: {exc!r}")
                continue
            if not isinstance(raw, str) or len(raw) <= 300:
                continue
            if selector == "#page_content":
                low = raw.lower()
                if "sales-history" not in low and "sell/history" not in low:
                    continue
            return raw
        return None

    while time.monotonic() < deadline:
        polls += 1
        got = try_extract()
        if got is not None:
            _detail(
                f"         Sale history DOM via CDP ({len(got) // 1024} KiB) after "
                f"{polls} poll(s); parsing …"
            )
            return _wrap_sale_history_html(got)
        now = time.monotonic()
        if now - last_log >= 6.0:
            _detail(
                f"         Waiting for sale-history in DOM (poll #{polls}, "
                f"CDP probe every {poll:.0f}s) …"
            )
            last_log = now
        time.sleep(poll)

    try:
        probe = driver.run_js(
            "return document.body ? document.body.innerHTML.length : -1"
        )
        _detail(f"         Debug: document.body.innerHTML length ≈ {probe!r}")
    except Exception as exc:
        _detail(f"         Debug: could not read body length: {exc!r}")

    _detail("         Blocking up to 12s for table.sales-history-table …")
    try:
        el = driver.wait_for_element("table.sales-history-table", wait=12)
        raw = el.html
        if isinstance(raw, str) and len(raw) > 200:
            _detail(f"         Sale history table ({len(raw) // 1024} KiB); parsing …")
            return _wrap_sale_history_html(raw)
    except Exception as exc:
        _detail(f"         Final table wait: {exc!r}")

    try:
        el = driver.wait_for_element("#page_content", wait=8)
        raw = el.html
        if isinstance(raw, str) and len(raw) > 200:
            low = raw.lower()
            if "sales-history" in low or "sell/history" in low:
                _detail(f"         #page_content ({len(raw) // 1024} KiB); parsing …")
                return _wrap_sale_history_html(raw)
    except Exception as exc:
        _detail(f"         Final #page_content wait: {exc!r}")

    _detail("         No extractable HTML — returning empty wrapper.")
    return _wrap_sale_history_html("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Discogs sale history pages into SQLite (Botasaurus + logged-in profile)",
    )
    parser.add_argument(
        "--release-ids",
        type=Path,
        required=True,
        help="Text file: one release ID per line (# comments ok); duplicates are ignored",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "Path to sale_history.sqlite. If relative, resolved from the **monorepo root** "
            "(parent of ``price_estimator/``). Example: ``price_estimator/data/cache/sale_history.sqlite``"
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help=(
            "Chrome user-data dir with Discogs login (default: env "
            "DISCOGS_SALE_HISTORY_BROWSER_PROFILE)"
        ),
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window (use for first-time Discogs login into the profile)",
    )
    parser.add_argument(
        "--login-wait-seconds",
        type=float,
        default=-1.0,
        metavar="SEC",
        help=(
            "After opening discogs.com before the first scrape: sleep this many seconds "
            "so you can log in. Default: 120 with --no-headless, 0 in headless. "
            "The script polls every 2s and continues as soon as you are signed in. "
            "Pass 0 to skip the wait even when --no-headless."
        ),
    )
    parser.add_argument(
        "--login-pause",
        action="store_true",
        help=(
            "After opening discogs.com: block until you press Enter in this terminal "
            "(unlimited time to log in). Overrides --login-wait-seconds for the initial wait."
        ),
    )
    parser.add_argument(
        "--assume-logged-in",
        action="store_true",
        help=(
            "Skip opening www.discogs.com and skip login polling. Use when your Chrome "
            "profile is already signed in but the homepage load hangs (e.g. Cloudflare). "
            "The script goes straight to sale-history URLs."
        ),
    )
    parser.add_argument(
        "--bypass-cloudflare",
        action="store_true",
        help=(
            "Pass bypass_cloudflare=True into every driver.get (old behavior). "
            "Can hang on normal Discogs. Default: get without bypass; if the tab title "
            'is exactly the Cloudflare interstitial (title is "Just a moment..."), '
            "run bypass once automatically."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds to wait between release pages (default: 4)",
    )
    parser.add_argument(
        "--resume-hours",
        type=float,
        default=0.0,
        help="With --resume, skip release_id if last successful fetch is newer than this many hours (0=only skip on any success)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs that already have a successful fetch recorded",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N successful scrapes (0 = no limit)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose stderr (heartbeats, DOM polls, navigation sub-steps). Default is compact.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=40,
        metavar="N",
        help=(
            "When not using --verbose, print one summary line every N queue positions "
            "(1..N) finished — counts skips and scrape outcomes. 0 disables. "
            "Individual errors still print immediately."
        ),
    )
    parser.add_argument(
        "--on-cloudflare",
        choices=("stop", "backoff"),
        default="stop",
        help=(
            "After navigation + parse, if a Cloudflare-style challenge is detected: "
            "'stop' exits the whole run immediately (exit code 3). "
            "'backoff' waits --cloudflare-backoff-seconds then retries navigation once "
            "for that release before giving up."
        ),
    )
    parser.add_argument(
        "--cloudflare-backoff-seconds",
        type=float,
        default=120.0,
        metavar="SEC",
        help="Used with --on-cloudflare backoff before the extra navigation retry.",
    )
    args = parser.parse_args()
    out.verbose = bool(args.verbose)
    _line_buffer_stdio()
    if args.assume_logged_in and args.login_pause:
        print(
            "Note: --assume-logged-in skips the homepage step, so --login-pause is ignored.",
            file=sys.stderr,
        )

    try:
        from shared.project_env import load_project_dotenv
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    load_project_dotenv()

    repo = _repo_root()
    db_path = args.db or (repo / "price_estimator" / "data" / "cache" / "sale_history.sqlite")
    if not db_path.is_absolute():
        db_path = (repo / db_path).resolve()

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
        print(
            f"Profile directory does not exist: {profile}\n"
            "Create it by running once with --no-headless and logging in.",
            file=sys.stderr,
        )
        return 1

    from botasaurus.browser import Driver
    from botasaurus_driver.user_agent import UserAgent
    from botasaurus_driver.window_size import WindowSize

    from price_estimator.src.scrape.discogs_sale_history_parse import (
        parse_sale_history_html,
        sale_history_url,
    )
    from price_estimator.src.storage.sale_history_db import SaleHistoryDB

    store = SaleHistoryDB(db_path)
    rp = args.release_ids.expanduser()
    if not rp.is_file():
        print(f"--release-ids not found: {rp}", file=sys.stderr)
        return 1

    ids = ordered_unique_release_ids(rp)
    n_ids = len(ids)
    if n_ids == 0:
        print(
            f"No numeric release IDs found in {rp} (lines must start with digits). "
            "Nothing to scrape.",
            file=sys.stderr,
        )
        return 1
    _progress(f"Queued {n_ids} release ID(s) from {rp.name} → {db_path}")
    _detail(
        "Progress on stderr; the browser navigates each sale-history URL. "
        "Omit --verbose for compact output."
    )

    # Botasaurus defaults wait_for_complete_page_load=True → driver.get blocks until
    # document.readyState === "complete". Discogs sale-history often stays "interactive"
    # while charts/analytics keep loading, so the UI looks done but get() would spin
    # (and our heartbeat would repeat) for a long time.
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

    if args.assume_logged_in:
        _detail(
            "--assume-logged-in: skipping www.discogs.com; going straight to sale-history URLs."
        )
    elif args.login_pause or login_wait > 0:
        try:
            if out.verbose:
                _progress("Opening https://www.discogs.com/ (log in if needed) …")
            else:
                _detail("Opening https://www.discogs.com/ …")
            _detail("Warmup uses direct get, 75s cap.")
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
                _progress(
                    "When you are logged in and ready, press Enter here to start scraping…"
                )
                try:
                    input()
                except EOFError:
                    _detail("(no stdin; continuing immediately)")
            else:
                _wait_for_discogs_login(driver, float(login_wait))
            _detail("Starting sale-history requests …")
        except Exception as e:
            print(f"Login warmup navigation failed: {e}", file=sys.stderr)
            try:
                driver.close()
            except Exception:
                pass
            return 1

    if not (args.login_pause or login_wait > 0) and not args.assume_logged_in:
        _detail(
            f"No login countdown (--login-wait-seconds 0 or headless); "
            f"starting {n_ids} sale-history page(s) immediately."
        )

    ok = skip = err = 0
    max_ok = args.max if args.max and args.max > 0 else None
    exit_code = 0

    def periodic_line(i_pos: int) -> None:
        pe = args.progress_every
        if out.verbose or pe <= 0 or i_pos % pe != 0:
            return
        _progress(f"Progress: ok={ok} skip={skip} err={err} ({i_pos}/{n_ids})")

    try:
        for i, rid in enumerate(ids, start=1):
            if max_ok is not None and ok >= max_ok:
                break
            if args.resume:
                if args.resume_hours and args.resume_hours > 0:
                    if store.should_skip_resume(rid, ok_hours=args.resume_hours):
                        skip += 1
                        if out.verbose:
                            _progress(f"[skip] {i}/{n_ids} release {rid} (resume)")
                        periodic_line(i)
                        continue
                else:
                    st = store.last_status(rid)
                    if st and st.get("status") == "ok":
                        skip += 1
                        if out.verbose:
                            _progress(
                                f"[skip] {i}/{n_ids} release {rid} (already ok in DB)"
                            )
                        periodic_line(i)
                        continue

            url = sale_history_url(rid)
            _detail(f"[{i}/{n_ids}] release {rid}")
            _detail(f"         {url}")
            _detail(
                "         driver.get (heartbeat only with --verbose). "
                'Cloudflare bypass only if title is "Just a moment..."; '
                "else use --bypass-cloudflare."
            )

            max_cf = 2 if args.on_cloudflare == "backoff" else 1
            cf_try = 0
            rid_done = False
            while cf_try < max_cf and not rid_done:
                cf_try += 1
                try:
                    hb_stop = threading.Event()
                    hb = threading.Thread(
                        target=_heartbeat_while,
                        args=(f"GET sale history {rid}", hb_stop),
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
                    _detail(
                        "         Navigation finished; polling DOM for sale table (no reloads) …"
                    )
                    html = _pull_sale_history_html(driver)
                    _detail(f"         HTML ready ({len(html) // 1024} KiB), parsing …")
                except Exception as e:
                    store.record_failure(rid, f"navigation: {e}")
                    err += 1
                    print(
                        f"[err] {i}/{n_ids} release {rid}: navigation failed: {e}",
                        file=sys.stderr,
                    )
                    rid_done = True
                    break

                parsed = parse_sale_history_html(html, rid)
                kind = _cloudflare_challenge_kind(driver)
                if _cloudflare_blocks_scrape(kind, html, parsed):
                    _cloudflare_stderr_banner(
                        kind=kind,
                        release_id=rid,
                        url=url,
                        hint_extra=(
                            "Exiting (--on-cloudflare stop). "
                            "Solve the challenge in the browser (--no-headless), then retry."
                            if args.on_cloudflare == "stop"
                            else (
                                f"Sleeping {args.cloudflare_backoff_seconds:.0f}s then "
                                "retrying navigation once for this release."
                            )
                        ),
                    )
                    if args.on_cloudflare == "stop":
                        store.record_failure(rid, f"cloudflare:{kind}")
                        err += 1
                        exit_code = 3
                        rid_done = True
                        break
                    if cf_try < max_cf:
                        time.sleep(max(1.0, args.cloudflare_backoff_seconds))
                        continue
                    store.record_failure(rid, f"cloudflare:{kind}")
                    err += 1
                    rid_done = True
                    break

                if looks_like_login_or_challenge(html):
                    store.record_failure(
                        rid,
                        "login_or_bot_challenge_page",
                        warnings=["expected_sale_history_after_login"],
                    )
                    err += 1
                    print(
                        f"[err] {i}/{n_ids} release {rid}: login/challenge page — refresh session in profile",
                        file=sys.stderr,
                    )
                    rid_done = True
                    break

                if (
                    parsed.parse_warnings
                    and "sales_table_not_found" in parsed.parse_warnings
                ):
                    store.record_failure(
                        rid,
                        "sales_table_not_found",
                        warnings=parsed.parse_warnings,
                    )
                    err += 1
                    print(
                        f"[err] {i}/{n_ids} release {rid}: could not find sales table",
                        file=sys.stderr,
                    )
                    rid_done = True
                    break

                store.upsert_parsed(parsed, status="ok", error=None)
                ok += 1
                if out.verbose:
                    _progress(
                        f"[ok] {i}/{n_ids} release {rid}: stored {len(parsed.rows)} sale row(s)"
                    )
                rid_done = True
                break

            if exit_code == 3:
                periodic_line(i)
                break

            if i < n_ids:
                d = max(0.5, args.delay)
                _detail(f"         Pausing {d:.1f}s before next release …")
            periodic_line(i)
            time.sleep(max(0.5, args.delay))
    finally:
        _detail("Closing browser …")
        try:
            driver.close()
        except Exception:
            pass

    if exit_code == 0:
        _progress(
            f"Finished. ok={ok} skipped={skip} errors={err} (of {n_ids} queued) → {db_path}"
        )
    else:
        _progress(
            f"Stopped early (exit {exit_code}, cloudflare). "
            f"ok={ok} skipped={skip} errors={err} ({n_ids} queued) → {db_path}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
