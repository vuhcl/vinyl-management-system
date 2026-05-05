"""
grader/src/data/ingest_discogs.py

Shim entrypoint for Discogs marketplace ingestion. Implementation lives in
:mod:`grader.src.data.discogs_ingest` (split modules, ≤500 LOC each).

Usage:
    python -m grader.src.data.ingest_discogs
    python -m grader.src.data.ingest_discogs --dry-run
"""

from __future__ import annotations

import logging

from grader.src.data.discogs_ingest import (
    DEFAULT_DISCOGS_FORMAT_FILTER,
    DiscogsIngester,
    RateLimiter,
    _DEFAULT_GENERIC_NOTE_PATTERNS,
    _DEFAULT_ITEM_SPECIFIC_HINTS,
    _DEFAULT_PRESERVATION_KEYWORDS,
    normalize_seller_comment_text,
)

__all__ = [
    "DEFAULT_DISCOGS_FORMAT_FILTER",
    "DiscogsIngester",
    "RateLimiter",
    "_DEFAULT_GENERIC_NOTE_PATTERNS",
    "_DEFAULT_ITEM_SPECIFIC_HINTS",
    "_DEFAULT_PRESERVATION_KEYWORDS",
    "normalize_seller_comment_text",
]


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Discogs marketplace ingestion")
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse without writing output or logging to MLflow",
    )
    parser.add_argument(
        "--target-per-grade",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override grader.yaml data.discogs.target_per_grade "
            "(max raw listings per canonical grade; lower = faster smoke ingest)"
        ),
    )
    parser.add_argument(
        "--format",
        default=None,
        metavar="NAME",
        help=(
            "Override data.discogs.format_filter (API format= + listing match). "
            f"Omit to use YAML or built-in default ({DEFAULT_DISCOGS_FORMAT_FILTER!r})."
        ),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Only use cached raw inventory pages already on disk. "
            "If a page is missing, skip it instead of calling Discogs."
        ),
    )
    args = parser.parse_args()

    ingester = DiscogsIngester(
        config_path=args.config,
        guidelines_path=args.guidelines,
        target_per_grade=args.target_per_grade,
        format_filter=args.format,
        cache_only=args.cache_only,
    )
    ingester.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
