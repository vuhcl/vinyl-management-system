#!/usr/bin/env python3
"""
Pick the preset key with highest test_mean_macro_f1 among the last N rows of
transformer_tune_results.csv (default N=6 for ``--presets all``).

Usage (repo root):
  PRESET=$(uv run python scripts/last_sweep_best_preset.py)
  uv run python -m grader.src.models.transformer_tune --config grader/configs/grader.yaml --promote "$PRESET"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default="grader/reports/transformer_tune_results.csv",
        help="Path to tuning results CSV",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=6,
        help="Number of trailing rows to treat as one full sweep",
    )
    args = p.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < args.last_n:
        raise SystemExit(
            f"Need at least {args.last_n} rows in {path}, got {len(rows)}"
        )
    tail = rows[-args.last_n :]
    best = max(tail, key=lambda r: float(r["test_mean_macro_f1"]))
    print(best["preset"])


if __name__ == "__main__":
    main()
