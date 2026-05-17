#!/usr/bin/env python3
"""CLI: export ``releases_features`` SQLite table to canonical Parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_estimator.src.monitoring.export_features import export_releases_features_to_parquet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sqlite",
        required=True,
        type=Path,
        help="Path to feature store SQLite",
    )
    p.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="Output Parquet path",
    )
    args = p.parse_args()
    n = export_releases_features_to_parquet(args.sqlite, args.parquet)
    print({"rows": n, "parquet": str(args.parquet)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
