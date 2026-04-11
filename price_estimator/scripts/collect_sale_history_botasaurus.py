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


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    """Progress to stderr so it still shows if something interferes with stdout."""
    print(msg, file=sys.stderr, flush=True)


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
        _progress(
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
                _progress(
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
            _progress(f"         … still busy ({label}), {n * interval:.0f}s elapsed …")
        elif n == max_lines + 1:
            _progress(
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
            _progress(
                "Signed in detected in the browser — continuing (no need to wait out the timer)."
            )
            return
        if not announced:
            _progress(
                f"Waiting for Discogs sign-in (checking every 2s, max {max_seconds:.0f}s; "
                "continues immediately once logged in) …"
            )
            announced = True
        now = time.monotonic()
        if now - last_status >= 12.0:
            left = max(0.0, deadline - now)
            extra = ""
            if state == "challenge":
                extra = " (Cloudflare challenge in progress — finish it in the browser) "
            elif state == "logged_out":
                extra = " (still on sign-in — log in in the browser) "
            _progress(f"         …{extra}~{left:.0f}s left before proceeding anyway …")
            last_status = now
        time.sleep(2.0)
    _progress(
        "Login poll window ended — proceeding. If you are not signed in, the next pages may fail."
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
                _progress(f"         {label}: CDP get_outer_html failed: {exc!r}")
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
            _progress(
                f"         Sale history DOM via CDP ({len(got) // 1024} KiB) after "
                f"{polls} poll(s); parsing …"
            )
            return _wrap_sale_history_html(got)
        now = time.monotonic()
        if now - last_log >= 6.0:
            _progress(
                f"         Waiting for sale-history in DOM (poll #{polls}, "
                f"CDP probe every {poll:.0f}s) …"
            )
            last_log = now
        time.sleep(poll)

    try:
        probe = driver.run_js(
            "return document.body ? document.body.innerHTML.length : -1"
        )
        _progress(f"         Debug: document.body.innerHTML length ≈ {probe!r}")
    except Exception as exc:
        _progress(f"         Debug: could not read body length: {exc!r}")

    _progress("         Blocking up to 12s for table.sales-history-table …")
    try:
        el = driver.wait_for_element("table.sales-history-table", wait=12)
        raw = el.html
        if isinstance(raw, str) and len(raw) > 200:
            _progress(f"         Sale history table ({len(raw) // 1024} KiB); parsing …")
            return _wrap_sale_history_html(raw)
    except Exception as exc:
        _progress(f"         Final table wait: {exc!r}")

    try:
        el = driver.wait_for_element("#page_content", wait=8)
        raw = el.html
        if isinstance(raw, str) and len(raw) > 200:
            low = raw.lower()
            if "sales-history" in low or "sell/history" in low:
                _progress(f"         #page_content ({len(raw) // 1024} KiB); parsing …")
                return _wrap_sale_history_html(raw)
    except Exception as exc:
        _progress(f"         Final #page_content wait: {exc!r}")

    _progress("         No extractable HTML — returning empty wrapper.")
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
        help="SQLite path (default: price_estimator/data/cache/sale_history.sqlite)",
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
    args = parser.parse_args()
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

    root = _root()
    db_path = args.db or (root / "data" / "cache" / "sale_history.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path

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
        looks_like_login_or_challenge,
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
    _progress(f"Queued {n_ids} release ID(s) from {rp} → SQLite {db_path}")
    _progress(
        "Progress lines below go to stderr; the browser will navigate each sale-history URL."
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
        _progress(
            "--assume-logged-in: skipping www.discogs.com (avoids long Cloudflare / "
            "google_get hangs). Going straight to sale-history pages."
        )
    elif args.login_pause or login_wait > 0:
        try:
            _progress(
                "Opening https://www.discogs.com/ — log in if needed, then continue… "
                "(using direct get, 75s cap — faster than google_get for warmup.)"
            )
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
                    _progress("(no stdin; continuing immediately)")
                _progress("Starting sale-history requests now (watch for [1/N] lines next).")
            else:
                _wait_for_discogs_login(driver, float(login_wait))
                _progress("Starting sale-history requests now (watch for [1/N] lines next).")
        except Exception as e:
            print(f"Login warmup navigation failed: {e}", file=sys.stderr)
            try:
                driver.close()
            except Exception:
                pass
            return 1

    if not (args.login_pause or login_wait > 0) and not args.assume_logged_in:
        _progress(
            f"No login countdown (--login-wait-seconds 0 or headless). "
            f"Starting {n_ids} sale-history page(s) immediately."
        )

    ok = skip = err = 0
    max_ok = args.max if args.max and args.max > 0 else None

    try:
        for i, rid in enumerate(ids, start=1):
            if max_ok is not None and ok >= max_ok:
                break
            if args.resume:
                if args.resume_hours and args.resume_hours > 0:
                    if store.should_skip_resume(rid, ok_hours=args.resume_hours):
                        skip += 1
                        _progress(f"[skip] {i}/{n_ids} release {rid} (resume)")
                        continue
                else:
                    st = store.last_status(rid)
                    if st and st.get("status") == "ok":
                        skip += 1
                        _progress(
                            f"[skip] {i}/{n_ids} release {rid} (already ok in DB)"
                        )
                        continue

            url = sale_history_url(rid)
            _progress(f"[{i}/{n_ids}] Navigating to sale history for release {rid} …")
            _progress(f"         {url}")
            _progress(
                "         One navigation (driver.get). Heartbeat every 10s only while "
                "get() is blocked. Cloudflare: auto-bypass only if title is "
                '"Just a moment..."; use --bypass-cloudflare to force in-get bypass.'
            )
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
                _progress(
                    "         Navigation finished; polling DOM for sale table (no reloads) …"
                )
                html = _pull_sale_history_html(driver)
                _progress(f"         HTML ready ({len(html) // 1024} KiB), parsing …")
            except Exception as e:
                store.record_failure(rid, f"navigation: {e}")
                err += 1
                print(f"[err] {i}/{n_ids} release {rid}: navigation failed: {e}", file=sys.stderr)
                time.sleep(args.delay)
                continue

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
                time.sleep(args.delay)
                continue

            parsed = parse_sale_history_html(html, rid)
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
            else:
                store.upsert_parsed(parsed, status="ok", error=None)
                ok += 1
                _progress(
                    f"[ok] {i}/{n_ids} release {rid}: stored {len(parsed.rows)} sale row(s)"
                )

            if i < n_ids:
                d = max(0.5, args.delay)
                _progress(f"         Pausing {d:.1f}s before next release …")
            time.sleep(max(0.5, args.delay))
    finally:
        _progress("Closing browser …")
        try:
            driver.close()
        except Exception:
            pass

    _progress(
        f"Finished. ok={ok} skipped={skip} errors={err} (of {n_ids} queued) → {db_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
