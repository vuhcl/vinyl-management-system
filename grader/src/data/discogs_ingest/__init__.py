"""Discogs inventory ingestion package (split from monolithic ingest_discogs)."""

from .constants import (
    DEFAULT_DISCOGS_FORMAT_FILTER,
    _DEFAULT_GENERIC_NOTE_PATTERNS,
    _DEFAULT_ITEM_SPECIFIC_HINTS,
    _DEFAULT_PRESERVATION_KEYWORDS,
)
from .ingester import DiscogsIngester
from .normalize import normalize_seller_comment_text
from .rate_limit import RateLimiter

__all__ = [
    "DEFAULT_DISCOGS_FORMAT_FILTER",
    "DiscogsIngester",
    "RateLimiter",
    "_DEFAULT_GENERIC_NOTE_PATTERNS",
    "_DEFAULT_ITEM_SPECIFIC_HINTS",
    "_DEFAULT_PRESERVATION_KEYWORDS",
    "normalize_seller_comment_text",
]
