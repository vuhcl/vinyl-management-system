#!/usr/bin/env python3
"""
Compute baseline ranking metrics (NDCG@K) for resume-ready comparisons.

Baselines (same evaluation split logic as recommender):
  - Random: random top-k over candidate items excluding train items
  - Popularity: top-k items by summed interaction strength in train

Outputs:
  - artifacts/baseline_ndcg.json (or --artifacts-dir)

Use ``--skip-random`` for popularity-only on large datasets (random baseline is slow because it
scores a random top-k over almost the full catalog per user).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute random + popularity baseline NDCG@K.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing interactions.parquet",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=120000,
        help="Sample this many interaction rows for speed (0 = use full dataset)",
    )
    parser.add_argument(
        "--min-interactions-per-user",
        type=int,
        default=2,
        help="Keep sampled users with at least this many interactions",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Compute NDCG@k",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for leave-one-out split + random baseline",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Where to write baseline_ndcg.json",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help=(
            "Only compute the popularity baseline (much faster on full data; "
            "random NDCG is omitted from the output)"
        ),
    )
    args = parser.parse_args()

    import numpy as np

    from recommender.src.data.preprocess import load_processed
    from recommender.src.evaluation.evaluate import leave_one_out_split
    from recommender.src.evaluation.metrics import ndcg_at_k

    interactions, _albums = load_processed(args.processed_dir)
    if interactions.empty:
        raise SystemExit("No interactions found in processed_dir.")

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
    sample["user_id"] = sample["user_id"].astype(str)
    sample["album_id"] = sample["album_id"].astype(str)

    train_int, test_int = leave_one_out_split(
        sample, random_state=args.random_state
    )

    # Relevant item(s) per user from test split.
    test_by_user: dict[str, set[str]] = (
        test_int.groupby("user_id")["album_id"].apply(set).to_dict()
    )

    # Train album ids per user (avoid O(users × rows) scans of train_int per user).
    _ti = train_int.assign(
        user_id=train_int["user_id"].astype(str),
        album_id=train_int["album_id"].astype(str),
    )
    train_by_user = _ti.groupby("user_id")["album_id"].apply(set).to_dict()

    # Candidate items are union of train/test items.
    all_items: np.ndarray = np.unique(
        np.concatenate(
            [
                train_int["album_id"].astype(str).values,
                test_int["album_id"].astype(str).values,
            ]
        )
    )

    rng = np.random.default_rng(args.random_state)

    def iter_users() -> Iterable[str]:
        return list(test_by_user.keys())

    def predict_random(
        user_id: str, train_items: set[str]
    ) -> list[str]:
        candidates = np.asarray(
            [it for it in all_items if it not in train_items], dtype=str
        )
        if len(candidates) == 0:
            return []
        n = min(args.k, len(candidates))
        choice = rng.choice(candidates, size=n, replace=False)
        return [str(x) for x in choice.tolist()]

    # Precompute popularity from train split.
    pop = (
        train_int.groupby("album_id")["strength"]
        .sum()
        .sort_values(ascending=False)
    )
    pop_items = pop.index.astype(str).values

    def predict_popularity(
        user_id: str, train_items: set[str]
    ) -> list[str]:
        out: list[str] = []
        for it in pop_items:
            if it in train_items:
                continue
            out.append(it)
            if len(out) >= args.k:
                break
        return out

    def eval_baseline(predict_fn) -> dict[str, float]:
        ndcgs: list[float] = []
        for uid in iter_users():
            relevant = test_by_user.get(uid) or set()
            if not relevant:
                continue

            train_items = train_by_user.get(uid, set())

            pred_items = predict_fn(uid, train_items)
            rel_arr = np.array(
                [1 if it in relevant else 0 for it in pred_items],
                dtype=np.int32,
            )
            ndcgs.append(ndcg_at_k(rel_arr, args.k))

        return {
            f"ndcg@{args.k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        }

    popularity_metrics = eval_baseline(
        lambda uid, train_items: predict_popularity(uid, train_items)
    )

    baselines: dict[str, dict[str, float]] = {}
    if not args.skip_random:
        baselines["random"] = eval_baseline(
            lambda uid, train_items: predict_random(uid, train_items)
        )
    baselines["popularity"] = popularity_metrics

    out = {
        "k": args.k,
        "split": {
            "sample_n": len(sample),
            "min_interactions_per_user": args.min_interactions_per_user,
            "random_state": args.random_state,
        },
        "baselines": baselines,
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.artifacts_dir / "baseline_ndcg.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {out_path}")
    print(json.dumps(out["baselines"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
