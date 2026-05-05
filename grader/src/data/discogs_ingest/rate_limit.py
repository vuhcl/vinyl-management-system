"""Token-bucket style rate limiter for Discogs API calls."""
from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Token bucket rate limiter.
    More robust than time.sleep — accounts for actual elapsed time
    so processing time between calls is not double-counted.
    """

    def __init__(self, calls_per_minute: int) -> None:
        self.min_interval: float = 60.0 / calls_per_minute
        self.last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self.last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_call = time.time()
