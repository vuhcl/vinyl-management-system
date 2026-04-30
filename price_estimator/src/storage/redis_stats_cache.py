"""Optional Redis layer in front of MarketplaceStatsDB.

The price API is read-heavy at the edge (the Chrome extension calls
``/estimate`` once per release page). Memorystore (Redis) absorbs those
hits without paying SQLite's per-request open/close cost and without
hammering Discogs on cold-cache misses.

SQLite remains the persistent source of truth: training labels, full
Discogs payloads, and every column on ``marketplace_stats`` continue
to live there. Redis caches **only** the inference response shape
returned by :meth:`InferenceService.fetch_stats` (a small JSON dict),
keyed by release id, with a configurable TTL (30 days by default to
match the demo system-design slide).

Failure mode: every Redis interaction is wrapped in a broad ``except``
that downgrades to a warning log and lets the caller fall through to
SQLite. Local dev (no ``REDIS_HOST`` set) and outages both end up at
the same SQLite-only behavior the service had before this layer.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
DEFAULT_KEY_PREFIX = "vinyliq:marketplace:stats:"


class RedisStatsCache:
    """Thin redis-py wrapper with graceful no-op when unavailable.

    Construction never raises: a missing ``redis`` package, an unset
    ``REDIS_HOST`` env var, or a failed ``PING`` all leave the cache in
    a disabled state where every method is a no-op. Use :meth:`enabled`
    to check the live state for logging/health output.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = DEFAULT_KEY_PREFIX,
    ) -> None:
        self.host = host if host is not None else os.environ.get("REDIS_HOST")
        try:
            self.port = int(os.environ.get("REDIS_PORT", port))
        except (TypeError, ValueError):
            self.port = port
        try:
            self.db = int(os.environ.get("REDIS_DB", db))
        except (TypeError, ValueError):
            self.db = db
        try:
            self.ttl_seconds = int(os.environ.get("REDIS_TTL_SECONDS", ttl_seconds))
        except (TypeError, ValueError):
            self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self._client: Any | None = None
        if self.host:
            self._client = self._connect()
        else:
            logger.info("REDIS_HOST not set; Redis stats cache disabled")

    def _connect(self) -> Any | None:
        try:
            # Lazy import keeps redis as a soft dependency: the service
            # still boots when the package is missing (e.g. pared-down
            # local environments). The Docker image always ships it.
            import redis as redis_pkg
        except ImportError:
            logger.info("redis package not installed; Redis stats cache disabled")
            return None
        try:
            client = redis_pkg.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                decode_responses=True,
            )
            client.ping()
            logger.info(
                "Redis stats cache connected (host=%s port=%d ttl=%ds)",
                self.host,
                self.port,
                self.ttl_seconds,
            )
            return client
        except Exception as e:
            # redis.exceptions.RedisError + socket errors all funnel here;
            # we never want a Memorystore hiccup to crash service init.
            logger.warning(
                "Redis connect failed (%s); falling back to SQLite only", e
            )
            return None

    def enabled(self) -> bool:
        return self._client is not None

    def _key(self, release_id: str) -> str:
        return f"{self.key_prefix}{str(release_id).strip()}"

    def get(self, release_id: str) -> dict[str, Any] | None:
        if self._client is None:
            return None
        try:
            raw = self._client.get(self._key(release_id))
        except Exception as e:
            logger.warning("Redis GET failed (%s)", e)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def set(self, release_id: str, payload: dict[str, Any]) -> None:
        if self._client is None:
            return
        try:
            self._client.setex(
                self._key(release_id),
                self.ttl_seconds,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
        except Exception as e:
            logger.warning("Redis SETEX failed (%s)", e)

    def invalidate(self, release_id: str) -> None:
        if self._client is None:
            return
        try:
            self._client.delete(self._key(release_id))
        except Exception as e:
            logger.warning("Redis DEL failed (%s)", e)
