"""Shared Botasaurus navigation helpers for Discogs website scrapers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

LogFn = Callable[[str], None]


def is_cloudflare_interstitial(driver: Any) -> bool:
    try:
        return (driver.title or "").strip() == "Just a moment..."
    except Exception:
        return False


def discogs_navigate(
    driver: Any,
    url: str,
    *,
    timeout: float,
    force_bypass_inside_get: bool,
    log: LogFn | None = None,
    skip_detect_bypass: bool = False,
) -> None:
    if force_bypass_inside_get:
        driver.get(url, bypass_cloudflare=True, timeout=timeout)
        return
    driver.get(url, bypass_cloudflare=False, timeout=timeout)
    if skip_detect_bypass:
        return
    if is_cloudflare_interstitial(driver):
        if log:
            log("         Cloudflare interstitial — detect_and_bypass_cloudflare once …")
        driver.detect_and_bypass_cloudflare()


def discogs_navigate_resilient(
    driver: Any,
    url: str,
    *,
    timeout: float,
    force_bypass_inside_get: bool,
    log: LogFn | None = None,
    skip_detect_bypass: bool = False,
) -> None:
    for attempt in (1, 2):
        try:
            discogs_navigate(
                driver,
                url,
                timeout=timeout,
                force_bypass_inside_get=force_bypass_inside_get,
                log=log,
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
                if log:
                    log(f"         Navigation CDP error ({e!r}); retrying once …")
                time.sleep(2.0)
                continue
            raise
