"""
Load ``listings_raw.ndjson`` from the release-listings Botasaurus output tree
and write ``discogs_release_marketplace.jsonl`` for harmonization.

Uses ``DiscogsIngester.parse_listing`` with ``offline_parse_only`` (no token).

Usage::

    PYTHONPATH=. python -m grader.src.data.ingest_discogs_release_marketplace \\
        --raw-dir grader/data/raw/discogs/release_marketplace
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from grader.src.config_io import load_yaml_mapping
from grader.src.data.ingest_discogs import DiscogsIngester

logger = logging.getLogger(__name__)


def _iter_listing_dicts(raw_dir: Path) -> Iterator[dict[str, Any]]:
    for nd in sorted(raw_dir.rglob("listings_raw.ndjson")):
        with open(nd, encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skip bad JSON %s:%d — %s",
                        nd,
                        line_num,
                        e,
                    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Discogs release-marketplace scrape NDJSON for harmonize"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="grader/configs/grader.yaml",
        help="grader.yaml path (repo root relative ok)",
    )
    parser.add_argument(
        "--guidelines",
        type=str,
        default=None,
        help="grading_guidelines.yaml (default: from grader.yaml guidelines_path)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("grader/data/raw/discogs/release_marketplace"),
        help="Root written by collect_discogs_release_listings_botasaurus",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (default: processed/discogs_release_marketplace.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count only; do not write output",
    )
    args = parser.parse_args()

    cfg = load_yaml_mapping(args.config)
    guidelines_path = args.guidelines or cfg.get(
        "guidelines_path",
        "grader/configs/grading_guidelines.yaml",
    )
    processed_dir = Path(cfg["paths"]["processed"])
    out_path = args.out or (processed_dir / "discogs_release_marketplace.jsonl")

    raw_dir = args.raw_dir
    if not raw_dir.is_dir():
        logger.error("raw-dir does not exist or is not a directory: %s", raw_dir)
        return 1

    ingester = DiscogsIngester(
        args.config,
        guidelines_path,
        offline_parse_only=True,
    )

    records: list[dict[str, Any]] = []
    n_in = 0
    for listing in _iter_listing_dicts(raw_dir):
        n_in += 1
        row = ingester.parse_listing(listing)
        if row:
            records.append(row)

    logger.info("Parsed %d raw listing(s) → %d kept", n_in, len(records))

    if args.dry_run:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
