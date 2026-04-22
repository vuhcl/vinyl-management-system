#!/usr/bin/env python3
"""
Hyperparameter tuning for recommender ALS (implicit).

This script tunes ALS parameters using your already-processed
`recommender/data/processed/interactions.parquet`, so it is fast enough to iterate and
does not re-run Discogs/Mongo ingestion.

It runs:
  1) sample interactions
  2) leave-one-out split (fixed random_state)
  3) fit ALS for each hyperparameter set
  4) compute NDCG@k / MAP@k / Recall@k

Outputs best config to:
  artifacts/als_tuning_best.json (or --artifacts-dir)
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune implicit ALS hyperparameters on real processed data."
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
        default=120000,
        help="Number of interaction rows to sample for tuning (0 = use full dataset)",
    )
    parser.add_argument(
        "--min-interactions-per-user",
        type=int,
        default=2,
        help="Only keep sampled users with at least this many interactions",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k used by NDCG/MAP/Recall",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for leave-one-out split",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default="16,32,64",
        help="Comma-separated list of factors to try",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        default="0.005,0.01,0.02",
        help="Comma-separated list of regularization values to try",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default="10,20,40",
        help="Comma-separated list of alpha values to try",
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default="1,2,3",
        help="Comma-separated list of ALS iterations to try",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=30,
        help="Cap number of runs (grid is truncated to this many)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Where to write als_tuning_best.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Project YAML (e.g. recommender/configs/base.yaml) for seed; "
            "use with processed data matching that pipeline"
        ),
    )
    args = parser.parse_args()

    from core.config import load_config
    from recommender.src.data.preprocess import load_processed
    from recommender.src.evaluation.evaluate import leave_one_out_split, run_evaluation

    project_cfg: dict | None = None
    if args.config is not None:
        project_cfg = load_config(args.config)
        if project_cfg.get("seed") is not None:
            args.random_state = int(project_cfg["seed"])

    def _parse_list(s: str) -> list[float]:
        out: list[float] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        return out

    factors = [int(x) for x in _parse_list(args.factors)]
    regularizations = _parse_list(args.regularization)
    alphas = _parse_list(args.alpha)
    iterations = [int(x) for x in _parse_list(args.iterations)]

    interactions, _albums = load_processed(Path(args.processed_dir))
    if interactions.empty:
        raise SystemExit("No interactions found in processed_dir.")

    if args.sample_n <= 0 or args.sample_n >= len(interactions):
        # Avoid an expensive `.sample(n=len(df))` shuffle when using full data.
        sample = interactions.copy()
    else:
        sample = interactions.sample(n=args.sample_n, random_state=args.random_state)

    sample_user_counts = sample["user_id"].value_counts()
    keep_users = sample_user_counts[
        sample_user_counts >= args.min_interactions_per_user
    ].index.astype(str)
    sample = sample[sample["user_id"].astype(str).isin(keep_users)].copy()
    sample["user_id"] = sample["user_id"].astype(str)
    sample["album_id"] = sample["album_id"].astype(str)

    train_int, test_int = leave_one_out_split(sample, random_state=args.random_state)

    grid = list(
        product(
            factors,
            regularizations,
            alphas,
            iterations,
        )
    )
    grid = grid[: args.max_runs]

    results: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    print("=== ALS tuning smoke ===")
    effective_sample_n = len(sample)
    print(f"sample rows: {effective_sample_n} users: {sample['user_id'].nunique()}")
    print(f"grid size (capped): {len(grid)}")
    if args.config and project_cfg:
        print(f"config: {args.config}")

    for run_idx, (factors_v, reg_v, alpha_v, it_v) in enumerate(grid, start=1):
        als_cfg = {
            "factors": factors_v,
            "regularization": reg_v,
            "alpha": alpha_v,
            "iterations": it_v,
        }
        metrics = run_evaluation(
            train_int,
            test_int,
            als_cfg,
            k=args.k,
            random_state=args.random_state,
            albums=None,
        )
        row = {"run": run_idx, "als_cfg": als_cfg, "metrics": metrics}
        results.append(row)

        score = metrics.get(f"ndcg@{args.k}", 0.0)
        if best is None or score > best.get("score", -1.0):  # type: ignore[operator]
            best = {"score": score, "als_cfg": als_cfg, "metrics": metrics}
        print(
            f"[{run_idx}/{len(grid)}] "
            f"factors={factors_v} reg={reg_v} alpha={alpha_v} it={it_v} "
            f"ndcg@{args.k}={metrics.get(f'ndcg@{args.k}', 0.0)}"
        )

    if best is None:
        raise SystemExit("No tuning runs executed.")

    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "als_tuning_best.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"\nBest config written to: {out_path}")
    print("Best:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
