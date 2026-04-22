#!/usr/bin/env python3
"""
Real-data smoke test for recommender ALS evaluation only.

This avoids the full `recommender.pipeline` (which can be slow on very
large datasets due to final ALS training + optional content similarity).

Steps:
  1) Load `recommender/data/processed/interactions.parquet`
  2) Sample interactions from real data
  3) Run leave-one-out split + `run_evaluation()` with tiny ALS params
  4) Print metrics and basic shapes
  5) Optional: ``--artifacts-dir artifacts`` writes JSON (default ``als_eval.json``)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path so `import recommender` works without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test ALS evaluation on real processed data.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("recommender/data/processed"),
        help="Directory containing interactions.parquet",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=200000,
        help="Number of interaction rows to sample (0 = use full dataset)",
    )
    parser.add_argument(
        "--min-interactions-per-user",
        type=int,
        default=2,
        help="Filter sampled users to those with >= this many interactions",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k for metrics",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=16,
        help="ALS latent factors",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="ALS iterations (keep small for smoke)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
        help="ALS regularization (lambda)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=20.0,
        help="ALS alpha (confidence weight)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for leave-one-out split and ALS (match baseline_ndcg.py)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="If set, write metrics JSON here (e.g. artifacts/)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="als_eval.json",
        help="Filename under artifacts-dir (default: als_eval.json)",
    )
    args = parser.parse_args()

    from recommender.src.data.preprocess import load_processed
    from recommender.src.evaluation.evaluate import leave_one_out_split, run_evaluation

    processed_dir = Path(args.processed_dir)
    interactions, _albums = load_processed(processed_dir)
    if interactions.empty:
        raise SystemExit("No interactions found in processed_dir.")

    # Keep it fast: sample rows first, then retain users with enough data.
    if args.sample_n <= 0 or args.sample_n >= len(interactions):
        sample = interactions.copy()
    else:
        sample = interactions.sample(
            n=min(args.sample_n, len(interactions)),
            random_state=args.random_state,
        )
    sample_user_counts = sample["user_id"].value_counts()
    keep_users = sample_user_counts[
        sample_user_counts >= args.min_interactions_per_user
    ].index.astype(str)
    sample = sample[sample["user_id"].astype(str).isin(keep_users)].copy()

    print("=== Smoke ALS eval (real processed data) ===")
    print("mode: full-catalog ALS recommend()")
    print("sample rows:", len(sample))
    print(
        "unique users:",
        sample["user_id"].nunique(),
    )
    print("unique items:", sample["album_id"].nunique())

    # Evaluation code expects string user_id/album_id.
    sample["user_id"] = sample["user_id"].astype(str)
    sample["album_id"] = sample["album_id"].astype(str)

    train_int, test_int = leave_one_out_split(sample, random_state=args.random_state)

    als_cfg = {
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "alpha": args.alpha,
    }

    metrics = run_evaluation(
        train_int,
        test_int,
        als_cfg,
        k=args.k,
        random_state=args.random_state,
        albums=None,
    )

    print("metrics:", metrics)

    if args.artifacts_dir is not None:
        out_dir = Path(args.artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.output_name
        payload = {
            "k": args.k,
            "mode": "full_catalog",
            "split": {
                "sample_n": len(sample),
                "min_interactions_per_user": args.min_interactions_per_user,
                "random_state": args.random_state,
                "unique_users": int(sample["user_id"].nunique()),
                "unique_items": int(sample["album_id"].nunique()),
            },
            "als": als_cfg,
            "metrics": metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
