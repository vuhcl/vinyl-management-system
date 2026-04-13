#!/usr/bin/env python3
"""
Collect Discogs marketplace data for a list of release IDs (labels + cache).

**Endpoints (see Discogs API docs):**

- ``GET /releases/{id}`` — catalog + ``community.want`` / ``community.have``,
  ``lowest_price``, ``num_for_sale`` (``curr_abbr`` for currency). In ``full``
  mode these populate the same SQLite listing columns that
  ``GET /marketplace/stats`` would (except ``blocked_from_sale``, which only
  exists on marketplace/stats).
- ``GET /marketplace/price_suggestions/{id}`` — **full** grade → suggested price map
  (all Discogs media conditions in one JSON object); **auth required**; returns
  ``{}`` if seller settings are incomplete. Stored verbatim in
  ``price_suggestions_json`` (empty responses do not overwrite a previously stored
  ladder in SQLite).
- ``GET /marketplace/stats/{id}`` — optional; used only in ``stats_only`` mode:
  lowest, ``num_for_sale``, ``blocked_from_sale`` (no true median in the API).

**Collect modes**

- ``full`` (default): ``/releases`` + ``/marketplace/price_suggestions`` (2 requests
  per id, each counted against ``--req-per-minute``). Listing scalars in SQLite
  come from the release response.
- ``stats_only``: legacy — only ``/marketplace/stats`` (1 request per id).

Loads repo-root ``.env`` automatically (``DISCOGS_TOKEN``, OAuth vars, etc.).

Parallel mode (default): multiple worker threads, **global** sliding-window
rate limit (Discogs ~60 req/min per token), retries on 429/5xx/timeouts.
IDs are read **streaming** from the file (safe for multi-million line lists).

Auth (first match):
  1. Personal token: ``DISCOGS_USER_TOKEN`` or ``DISCOGS_TOKEN``
  2. OAuth 1.0a: consumer key/secret + ``DISCOGS_OAUTH_TOKEN`` + secret

One-time OAuth:
  PYTHONPATH=. python price_estimator/scripts/collect_marketplace_stats.py --oauth-login

Examples:
  PYTHONPATH=. python price_estimator/scripts/collect_marketplace_stats.py \\
      --release-ids dump_release_ids.txt --workers 12 --req-per-minute 55

  PYTHONPATH=. python price_estimator/scripts/collect_marketplace_stats.py \\
      --collect-mode stats_only --release-ids dump_release_ids.txt --resume

  PYTHONPATH=. python price_estimator/scripts/collect_marketplace_stats.py \\
      --collect-mode full --curr-abbr USD --release-ids queue_shard_ab.txt --resume
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time

import requests
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from functools import partial
from typing import Callable, Iterator


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_personal_token(raw: str) -> str:
    """Strip BOM, outer quotes, and whitespace from a pasted token."""
    s = raw.replace("\ufeff", "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def iter_release_ids_from_file(path: Path) -> Iterator[str]:
    """Stream numeric release IDs (one per line); does not load whole file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            token = s.split()[0]
            if token.isdigit():
                yield token


class SlidingWindowLimiter:
    """
    At most *max_calls* acquisitions per *period_seconds* (global, thread-safe).
    """

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.period = float(period_seconds)
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            sleep_for: float = 0.0
            with self._lock:
                now = time.monotonic()
                while self._times and self._times[0] <= now - self.period:
                    self._times.popleft()
                if len(self._times) < self.max_calls:
                    self._times.append(now)
                    return
                sleep_for = self.period - (now - self._times[0]) + 0.02
            time.sleep(max(sleep_for, 0.02))


_tls_client = threading.local()
_tls_store: threading.local = threading.local()


def _thread_discogs_client():
    from shared.discogs_api.client import discogs_client_from_env

    if getattr(_tls_client, "client", None) is None:
        c = discogs_client_from_env()
        if c is None or not c.is_authenticated:
            raise RuntimeError("Discogs client not configured")
        _tls_client.client = c
    return _tls_client.client


def _thread_store(db_path: Path):
    from price_estimator.src.storage.marketplace_db import MarketplaceStatsDB

    if getattr(_tls_store, "store", None) is None:
        _tls_store.store = MarketplaceStatsDB(db_path)
    return _tls_store.store


def _process_one_rid(
    rid: str,
    *,
    db_path: Path,
    limiter: SlidingWindowLimiter,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    timeout: float,
    collect_mode: str = "full",
    curr_abbr: str | None = None,
) -> None:
    client = _thread_discogs_client()
    store = _thread_store(db_path)

    if collect_mode == "stats_only":
        limiter.acquire()
        raw = client.get_marketplace_stats_with_retries(
            rid,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
        )
        if not isinstance(raw, dict):
            raw = {}
        store.upsert(rid, raw)
        return

    release_pl: dict | None = None
    sugg_pl: dict | None = None

    limiter.acquire()
    try:
        release_pl = client.get_release_with_retries(
            rid,
            curr_abbr=curr_abbr,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
        )
    except requests.RequestException:
        release_pl = None
    if not isinstance(release_pl, dict):
        release_pl = None

    limiter.acquire()
    try:
        sugg_pl = client.get_price_suggestions_with_retries(
            rid,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
        )
    except requests.RequestException:
        sugg_pl = {}
    if not isinstance(sugg_pl, dict):
        sugg_pl = {}

    store.upsert(
        rid,
        {},
        release_payload=release_pl,
        price_suggestions_payload=sugg_pl,
    )


def _run_parallel(
    *,
    id_source: Iterator[str],
    db_path: Path,
    workers: int,
    req_per_minute: float,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    timeout: float,
    max_new: int | None,
    should_skip: Callable[[str], bool],
    progress_every: int,
    collect_mode: str = "full",
    curr_abbr: str | None = None,
) -> tuple[int, int, int]:
    """
    Returns (fetched_ok, skipped_filter, errors).
    """
    period = 60.0
    max_calls = max(1, int(req_per_minute))
    limiter = SlidingWindowLimiter(max_calls=max_calls, period_seconds=period)

    st = {"ok": 0, "skip": 0, "err": 0}
    st_lock = threading.Lock()
    stop = threading.Event()

    inflight = max(workers * 8, workers)
    ex = ThreadPoolExecutor(max_workers=workers)
    futures: set = set()

    def pump_completed(*, block: bool) -> None:
        nonlocal futures
        if not futures:
            return
        timeout_t = None if block else 0.5
        done, not_done = wait(futures, timeout=timeout_t, return_when=FIRST_COMPLETED)
        futures = set(not_done)
        for fut in done:
            try:
                fut.result()
                with st_lock:
                    st["ok"] += 1
                    if (
                        max_new is not None
                        and st["ok"] >= max_new
                        and not stop.is_set()
                    ):
                        stop.set()
                    if progress_every and st["ok"] % progress_every == 0:
                        print(
                            f"... ok={st['ok']} skip={st['skip']} err={st['err']}",
                            flush=True,
                        )
            except Exception as e:
                with st_lock:
                    st["err"] += 1
                print(f"worker error: {e}", file=sys.stderr)

    try:
        for rid in id_source:
            if stop.is_set():
                break
            if max_new is not None:
                with st_lock:
                    if st["ok"] >= max_new:
                        break
            if should_skip(rid):
                with st_lock:
                    st["skip"] += 1
                continue
            fut = ex.submit(
                partial(
                    _process_one_rid,
                    db_path=db_path,
                    limiter=limiter,
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                    backoff_max=backoff_max,
                    timeout=timeout,
                    collect_mode=collect_mode,
                    curr_abbr=curr_abbr,
                ),
                rid,
            )
            futures.add(fut)
            while len(futures) >= inflight:
                pump_completed(block=True)
                if stop.is_set():
                    break
            if stop.is_set():
                break

        while futures:
            pump_completed(block=True)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)

    return st["ok"], st["skip"], st["err"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Discogs release + marketplace data into SQLite (parallel + limiter)",
    )
    parser.add_argument(
        "--collect-mode",
        choices=("full", "stats_only"),
        default="full",
        help=(
            "full=/releases + /marketplace/price_suggestions (default, 2 req/id). "
            "stats_only=legacy /marketplace/stats only (1 req/id)."
        ),
    )
    parser.add_argument(
        "--curr-abbr",
        default=None,
        metavar="USD",
        help="Optional currency for GET /releases (e.g. USD, EUR).",
    )
    parser.add_argument(
        "--release-ids",
        type=Path,
        default=None,
        help="Text file: one release ID per line (# comments ok), streamed",
    )
    parser.add_argument(
        "--oauth-login",
        action="store_true",
        help="Run Discogs OAuth; print DISCOGS_OAUTH_* lines for .env",
    )
    parser.add_argument(
        "--personal-token",
        default=None,
        metavar="TOKEN",
        help=(
            "Personal access token for this run only (sets DISCOGS_USER_TOKEN after "
            ".env load, so it wins over .env). Avoids shell quoting issues."
        ),
    )
    parser.add_argument(
        "--personal-token-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Read token from file (first non-empty line, stripped). Same override "
            "behavior as --personal-token. Prefer chmod 600 on the file."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to marketplace_stats.sqlite",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread pool size (default: 8)",
    )
    parser.add_argument(
        "--req-per-minute",
        type=float,
        default=55.0,
        help="Global max completed HTTP starts per 60s window (default: 55)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per release for 429/5xx/network (default: 8)",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=1.5,
        help="Exponential backoff base seconds (default: 1.5)",
    )
    parser.add_argument(
        "--backoff-max",
        type=float,
        default=120.0,
        help="Backoff cap seconds (default: 120)",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=45.0,
        help="Per-request timeout seconds (default: 45)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N successes (0=quiet; default: 500)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip release_id already in the DB",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("memory", "query"),
        default="memory",
        help=(
            "memory=load all DB keys (fast skip, high RAM). "
            "query=SQLite EXISTS per ID (low RAM, slower skip). "
            "Use query for very large DBs."
        ),
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N successful new upserts (0 = no limit)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=(
            "Deprecated: if set (and you omit tuning), sets "
            "req-per-minute ≈ 60/(delay+0.25) for parity with old script"
        ),
    )
    args = parser.parse_args()

    try:
        from shared.discogs_api.client import discogs_client_from_env
        from shared.discogs_api.oauth1 import run_interactive_oauth
        from shared.project_env import load_project_dotenv
    except ImportError:
        print(
            "PYTHONPATH must include repo root (shared.discogs_api).",
            file=sys.stderr,
        )
        return 1

    load_project_dotenv()

    if args.personal_token and args.personal_token_file:
        print(
            "Use only one of --personal-token or --personal-token-file.",
            file=sys.stderr,
        )
        return 1
    if args.personal_token_file is not None:
        fp = args.personal_token_file.expanduser()
        if not fp.is_file():
            print(f"Token file not found: {fp}", file=sys.stderr)
            return 1
        raw = fp.read_text(encoding="utf-8", errors="replace")
        line = ""
        for ln in raw.splitlines():
            s = _normalize_personal_token(ln)
            if s and not s.startswith("#"):
                line = s
                break
        if not line:
            print("Token file has no non-empty line.", file=sys.stderr)
            return 1
        os.environ["DISCOGS_USER_TOKEN"] = line
    elif args.personal_token:
        tok = _normalize_personal_token(str(args.personal_token))
        if not tok:
            print("--personal-token is empty.", file=sys.stderr)
            return 1
        os.environ["DISCOGS_USER_TOKEN"] = tok

    if args.oauth_login:
        ck = (os.environ.get("DISCOGS_CONSUMER_KEY") or "").strip()
        cs = (os.environ.get("DISCOGS_CONSUMER_SECRET") or "").strip()
        if not ck or not cs:
            print(
                "Set DISCOGS_CONSUMER_KEY and DISCOGS_CONSUMER_SECRET in .env",
                file=sys.stderr,
            )
            return 1
        try:
            at, ats = run_interactive_oauth(ck, cs)
        except Exception as e:
            print(f"OAuth failed: {e}", file=sys.stderr)
            return 1
        print(
            "\nAdd these to your repo-root .env, then run again without "
            "--oauth-login:\n",
            flush=True,
        )
        print(f"DISCOGS_OAUTH_TOKEN={at}")
        print(f"DISCOGS_OAUTH_TOKEN_SECRET={ats}")
        return 0

    if not args.release_ids:
        print(
            "--release-ids is required unless using --oauth-login",
            file=sys.stderr,
        )
        return 1

    probe = discogs_client_from_env()
    if probe is None or not probe.is_authenticated:
        print(
            "No Discogs credentials: set DISCOGS_USER_TOKEN / DISCOGS_TOKEN in .env, "
            "or pass --personal-token / --personal-token-file, or OAuth "
            "(DISCOGS_OAUTH_* + consumer key/secret). Run --oauth-login once for OAuth.",
            file=sys.stderr,
        )
        return 1

    def _redact_discogs_error(msg: str) -> str:
        """Do not echo ``?token=`` query strings into logs."""
        return re.sub(
            r"([?&])token=[^&\s]+",
            r"\1token=<redacted>",
            msg,
            flags=re.IGNORECASE,
        )

    try:
        # Same class of call as ``full`` mode (personal token + OAuth both supported).
        probe.get_release("1")
    except requests.HTTPError as e:
        sc = e.response.status_code if e.response is not None else None
        if sc == 429:
            # 429 means Discogs accepted the request; bucket is just full (tests + collectors).
            print(
                "Discogs auth probe: 429 Too Many Requests (token is valid; wait or lower "
                "parallelism). Starting collection — workers will retry 429s.",
                flush=True,
            )
        elif sc == 401:
            hint = ""
            pt = (
                os.environ.get("DISCOGS_USER_TOKEN")
                or os.environ.get("DISCOGS_TOKEN")
                or ""
            ).strip()
            if pt:
                hint = f" (token length={len(pt)}; regenerate at Discogs → Settings → Developers)"
            print(
                "Discogs authentication failed (401): "
                f"{_redact_discogs_error(str(e))}{hint}",
                file=sys.stderr,
            )
            return 1
        else:
            print(
                f"Discogs auth probe failed: {_redact_discogs_error(str(e))}",
                file=sys.stderr,
            )
            return 1
    except Exception as e:
        print(
            "Discogs auth probe failed: "
            f"{_redact_discogs_error(str(e))}",
            file=sys.stderr,
        )
        return 1

    root = _root()
    db_path = args.db or (root / "data" / "cache" / "marketplace_stats.sqlite")
    if not db_path.is_absolute():
        db_path = root / db_path

    from price_estimator.src.storage.marketplace_db import MarketplaceStatsDB

    store = MarketplaceStatsDB(db_path)

    req_per_minute = float(args.req_per_minute)
    if args.delay is not None and args.delay > 0:
        req_per_minute = 60.0 / (args.delay + 0.25)
        print(
            f"Using --delay legacy mapping: req-per-minute={req_per_minute:.2f}",
            flush=True,
        )

    existing: set[str] | None = None
    if args.resume:
        if args.resume_mode == "memory":
            n_existing = store.count_rows()
            if n_existing > 1_500_000:
                print(
                    f"Resume memory mode loads {n_existing} IDs into RAM. "
                    "Consider --resume-mode query for large DBs.",
                    file=sys.stderr,
                )
            existing = store.existing_release_ids()

    def should_skip(rid: str) -> bool:
        if not args.resume:
            return False
        if existing is not None:
            return rid in existing
        return store.has_release_id(rid)

    max_new = args.max if args.max and args.max > 0 else None

    id_iter = iter_release_ids_from_file(args.release_ids)

    ca = args.curr_abbr.strip().upper() if args.curr_abbr else None

    print(
        f"Collecting marketplace data → {db_path} "
        f"(mode={args.collect_mode}, workers={args.workers}, "
        f"req/min≈{req_per_minute:.1f}, resume={args.resume}/{args.resume_mode})",
        flush=True,
    )

    ok, skipped, err = _run_parallel(
        id_source=id_iter,
        db_path=db_path,
        workers=max(1, args.workers),
        req_per_minute=req_per_minute,
        max_retries=max(0, args.max_retries),
        backoff_base=args.backoff_base,
        backoff_max=args.backoff_max,
        timeout=args.http_timeout,
        max_new=max_new,
        should_skip=should_skip,
        progress_every=max(0, args.progress_every),
        collect_mode=args.collect_mode,
        curr_abbr=ca,
    )

    msg = f"Done. fetched={ok} skipped={skipped} errors={err} db={db_path}"
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
