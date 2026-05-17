#!/usr/bin/env python3
"""
Smoke benchmark for Discogs ingest (dry-run + cache-only; no writes).

Run from repo root:
  uv run python grader/scripts/bench_ingest_smoke.py --dry-run --cache-only

Real throughput benchmarks need DISCOGS_TOKEN and are local-only; see
grader/docs/benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Smoke DiscogsIngester (timing wrapper)")
    p.add_argument("--config", default="grader/configs/grader.yaml")
    p.add_argument("--guidelines", default="grader/configs/grading_guidelines.yaml")
    p.add_argument("--dry-run", action="store_true", help="No file write / MLflow")
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Only read cached raw pages (skip missing without API)",
    )
    args = p.parse_args()
    if not Path(args.config).is_file():
        print(f"Missing config: {args.config}")
        return 1

    from grader.src.data.ingest_discogs import DiscogsIngester

    ingester = DiscogsIngester(
        config_path=args.config,
        guidelines_path=args.guidelines,
        cache_only=args.cache_only,
    )
    t0 = time.perf_counter()
    _ = ingester.run(dry_run=args.dry_run)
    elapsed = time.perf_counter() - t0
    print(f"ingest_discogs_smoke wall_s={elapsed:.4f} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
