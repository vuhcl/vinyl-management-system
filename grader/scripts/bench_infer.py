#!/usr/bin/env python3
"""
Wall-clock benchmark for Pipeline.predict_batch (cold lazy-load path).

Run from repo root:
  uv run python grader/scripts/bench_infer.py
  uv run python grader/scripts/bench_infer.py --config grader/configs/grader.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path


# Short fixed texts (≥3) — keep in-script for reproducibility.
_FIXTURE_TEXTS: tuple[str, ...] = (
    "NM sleeve, plays perfectly, no marks",
    "VG+ vinyl, light sleeve wear, cleaned ultrasonically",
    "Sealed copy, minor corner bump on shrink",
)


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Benchmark Pipeline.predict_batch")
    p.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader.yaml",
    )
    p.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    args = p.parse_args()
    cfg = Path(args.config)
    if not cfg.is_file():
        print(f"Missing config: {cfg}")
        return 1

    # Import after argparse so --help stays fast.
    from grader.src.pipeline import Pipeline

    pipe = Pipeline(
        config_path=str(cfg),
        guidelines_path=str(args.guidelines),
    )
    t0 = time.perf_counter()
    _ = pipe.predict_batch(list(_FIXTURE_TEXTS))
    elapsed = time.perf_counter() - t0
    print(f"predict_batch_n={len(_FIXTURE_TEXTS)} wall_s={elapsed:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
