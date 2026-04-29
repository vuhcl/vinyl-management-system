#!/usr/bin/env python3
"""
Pilot metrics for release-marketplace scrape → grader JSONL.

Reads processed records (``discogs_release_marketplace.jsonl`` or similar),
runs ``Preprocessor.compute_description_quality`` with ``grader.yaml``, and prints:

- total rows
- ``adequate_for_training`` count and rate (non–thin-note by current config)

Use after ``ingest_discogs_release_marketplace`` and before harmonize, or on
``unified.jsonl`` subsets.

  PYTHONPATH=. python price_estimator/scripts/report_grader_release_listings_pilot.py \\
      --jsonl grader/data/processed/discogs_release_marketplace.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adequate-rate pilot stats for grader JSONL",
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="Processed JSONL path")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("grader/configs/grader.yaml"),
        help="grader.yaml",
    )
    parser.add_argument(
        "--guidelines",
        type=Path,
        default=None,
        help="grading_guidelines.yaml (default from grader.yaml)",
    )
    args = parser.parse_args()

    if not args.jsonl.is_file():
        print(f"Not found: {args.jsonl}", file=sys.stderr)
        return 1

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gpath = args.guidelines or Path(
        cfg.get("guidelines_path", "grader/configs/grading_guidelines.yaml")
    )

    from grader.src.data.preprocess import Preprocessor

    pre = Preprocessor(str(args.config), str(gpath))

    n = 0
    n_ok = 0
    with open(args.jsonl, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or ""
            clean = pre.clean_text(text)
            dq = pre.compute_description_quality(text, clean)
            n += 1
            if dq.get("adequate_for_training"):
                n_ok += 1

    rate = (100.0 * n_ok / n) if n else 0.0
    print(f"rows={n} adequate_for_training={n_ok} rate={rate:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
