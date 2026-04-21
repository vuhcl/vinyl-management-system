#!/usr/bin/env python3
"""
ALS full-catalog hit-rate diagnostic.

Answers: "If I pass ALS top-N candidates to the reranker, what fraction of
test users' held-out items fall within that top-N?"

Method: leave-one-out split on processed interactions; fit ALS on the train
split only (so the held-out item is unseen during training); for each test
user, rank all ~90k items by the user x item latent dot product, exclude
train items, find the rank of the held-out item; aggregate hit_rate@N for a
list of cutoffs.

Use the output to calibrate ``reranker.candidate_top_n`` in
``recommender/configs/base.yaml`` (target ``hit_rate@N >= 0.8``).

Writes ``artifacts/als_hit_rate.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_cutoffs(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure ALS full-catalog hit_rate@N to calibrate reranker "
            "candidate_top_n."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Project YAML (e.g. recommender/configs/base.yaml). ALS hyperparams "
            "are read from ``als.*`` unless overridden on the CLI."
        ),
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
        default=0,
        help="Sample N interaction rows for speed (0 = use full dataset)",
    )
    parser.add_argument(
        "--min-interactions-per-user",
        type=int,
        default=2,
        help="Keep sampled users with at least this many interactions",
    )
    parser.add_argument(
        "--cutoffs",
        type=str,
        default="50,100,200,500,1000,2000,5000",
        help="Comma-separated list of top-N cutoffs for hit_rate@N",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for leave-one-out split (match pipeline seed for comparability)",
    )
    parser.add_argument("--factors", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--regularization", type=float, default=None)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Where to write als_hit_rate.json",
    )
    args = parser.parse_args()

    import numpy as np

    from core.config import load_config
    from recommender.src.data.preprocess import load_processed
    from recommender.src.evaluation.evaluate import leave_one_out_split
    from recommender.src.features.build_matrix import (
        build_user_item_matrix,
        get_user_item_mappers,
    )
    from recommender.src.models.als import train_als

    project_cfg: dict = {}
    if args.config is not None:
        project_cfg = load_config(args.config) or {}
        if project_cfg.get("seed") is not None:
            args.random_state = int(project_cfg["seed"])

    als_defaults = (project_cfg.get("als") or {}) if project_cfg else {}

    def _pick(cli_val, yaml_key: str, fallback):
        if cli_val is not None:
            return cli_val
        return als_defaults.get(yaml_key, fallback)

    als_cfg = {
        "factors": int(_pick(args.factors, "factors", 64)),
        "iterations": int(_pick(args.iterations, "iterations", 15)),
        "alpha": float(_pick(args.alpha, "alpha", 10.0)),
        "regularization": float(
            _pick(args.regularization, "regularization", 0.01)
        ),
    }
    cutoffs = _parse_cutoffs(args.cutoffs)
    if not cutoffs:
        raise SystemExit("--cutoffs must contain at least one positive integer")

    interactions, _albums = load_processed(Path(args.processed_dir))
    if interactions.empty:
        raise SystemExit("No interactions found in processed_dir.")

    if args.sample_n <= 0 or args.sample_n >= len(interactions):
        sample = interactions.copy()
    else:
        sample = interactions.sample(n=args.sample_n, random_state=args.random_state)
    user_counts = sample["user_id"].value_counts()
    keep_users = user_counts[
        user_counts >= args.min_interactions_per_user
    ].index.astype(str)
    sample = sample[sample["user_id"].astype(str).isin(keep_users)].copy()
    sample["user_id"] = sample["user_id"].astype(str)
    sample["album_id"] = sample["album_id"].astype(str)

    train_int, test_int = leave_one_out_split(sample, random_state=args.random_state)

    all_item_ids = np.unique(
        np.concatenate(
            [
                train_int["album_id"].astype(str).values,
                test_int["album_id"].astype(str).values,
            ]
        )
    )
    matrix, user_ids, item_ids = build_user_item_matrix(
        train_int, weight_col="strength", all_item_ids=all_item_ids
    )
    user_id2idx, item_id2idx, _, _ = get_user_item_mappers(user_ids, item_ids)

    print("=== ALS hit-rate diagnostic ===")
    print(f"sample rows: {len(sample)} users: {sample['user_id'].nunique()}")
    print(f"items (matrix cols): {len(item_ids)}")
    print(f"als_cfg: {als_cfg}")
    print(f"cutoffs: {cutoffs}")

    t0 = time.time()
    model = train_als(
        matrix,
        factors=als_cfg["factors"],
        regularization=als_cfg["regularization"],
        iterations=als_cfg["iterations"],
        alpha=als_cfg["alpha"],
        random_state=args.random_state,
    )
    fit_s = time.time() - t0
    print(f"ALS fit time: {fit_s:.1f}s")

    user_factors = np.asarray(model.user_factors, dtype=np.float64)
    item_factors = np.asarray(model.item_factors, dtype=np.float64)

    train_by_user: dict[str, set[str]] = (
        train_int.groupby("user_id")["album_id"].apply(set).to_dict()
    )
    test_by_user: dict[str, set[str]] = (
        test_int.groupby("user_id")["album_id"].apply(set).to_dict()
    )
    item_train_counts: dict[str, int] = {
        str(a): int(v)
        for a, v in train_int.groupby("album_id", sort=False).size().items()
    }

    ranks: list[int] = []
    heldout_pop: list[int] = []
    users_evaluated = 0
    users_skipped_no_user = 0
    users_skipped_no_item = 0

    t0 = time.time()
    for uid, held in test_by_user.items():
        if uid not in user_id2idx:
            users_skipped_no_user += 1
            continue
        held_ids = [a for a in held if a in item_id2idx]
        if not held_ids:
            users_skipped_no_item += 1
            continue
        u_idx = user_id2idx[uid]
        scores = item_factors @ user_factors[u_idx]
        train_idxs = [
            item_id2idx[a]
            for a in train_by_user.get(uid, set())
            if a in item_id2idx
        ]
        if train_idxs:
            scores[np.asarray(train_idxs, dtype=np.int64)] = -np.inf

        rel_idx = item_id2idx[held_ids[0]]
        rel_score = scores[rel_idx]
        if not np.isfinite(rel_score):
            users_skipped_no_item += 1
            continue
        rank = int(np.sum(scores > rel_score)) + 1
        ranks.append(rank)
        heldout_pop.append(int(item_train_counts.get(held_ids[0], 0)))
        users_evaluated += 1

    rank_s = time.time() - t0
    print(f"rank scan time: {rank_s:.1f}s over {users_evaluated} users")

    if users_evaluated == 0:
        raise SystemExit("No test users had a scorable held-out item.")

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    hit_rates = {
        f"hit_rate@{n}": float(np.mean(ranks_arr <= n)) for n in cutoffs
    }

    print("\n  N      hit_rate")
    for n in cutoffs:
        print(f"  {n:>5d}    {hit_rates[f'hit_rate@{n}']:.4f}")

    median_pop = float(np.median(heldout_pop)) if heldout_pop else 0.0
    mean_rank = float(np.mean(ranks_arr))
    median_rank = float(np.median(ranks_arr))

    print(
        f"\nmean held-out rank: {mean_rank:.1f}  median: {median_rank:.1f}"
        f"  median held-out train count: {median_pop:.1f}"
    )

    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "als": als_cfg,
        "split": {
            "sample_n": int(len(sample)),
            "min_interactions_per_user": int(args.min_interactions_per_user),
            "random_state": int(args.random_state),
            "users_evaluated": int(users_evaluated),
            "users_skipped_no_user_factor": int(users_skipped_no_user),
            "users_skipped_no_item_factor": int(users_skipped_no_item),
        },
        "cutoffs": cutoffs,
        "hit_rates": hit_rates,
        "rank_stats": {
            "mean": mean_rank,
            "median": median_rank,
            "median_heldout_train_count": median_pop,
        },
        "timing_s": {
            "als_fit": float(fit_s),
            "rank_scan": float(rank_s),
        },
    }
    out_path = out_dir / "als_hit_rate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
