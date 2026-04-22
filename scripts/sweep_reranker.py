#!/usr/bin/env python3
"""
Sweep reranker hyperparameters on top of a single fitted ALS model.

ALS is trained once. For each grid point the reranker training frame and
model are rebuilt, then reranked evaluation is run. Both reranker-on and
ALS-only stratified (pop_head / pop_tail) metrics are preserved per run so
the winner rule can compare them directly.

Winner rule (printed at end): among runs with
  ndcg@k > als_only_ndcg@k  AND  ndcg@k_pop_tail > als_only_ndcg@k_pop_tail
pick the one with the smallest |als_only_ndcg@k_pop_head - ndcg@k_pop_head|.
If no run satisfies the criteria, report that the reranker should stay
disabled at current defaults.

Outputs artifacts/sweep_reranker.json with full per-run metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep reranker hyperparameters on top of a single fitted ALS "
            "model. Preserves ALS-only pop_head/pop_tail per run for the "
            "winner rule."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("recommender/configs/base.yaml"),
        help="Project YAML; als.* block is used to fit ALS once.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing interactions.parquet + albums.parquet.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=3_000_000,
        help="Interaction rows to sample (0 = use full dataset).",
    )
    parser.add_argument(
        "--min-interactions-per-user",
        type=int,
        default=2,
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-types",
        type=str,
        default="linear,pointwise",
        help="Comma-separated: linear and/or pointwise.",
    )
    parser.add_argument(
        "--candidate-top-n",
        type=str,
        default="200,500,1000",
        help="Grid of reranker.candidate_top_n.",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=str,
        default="0.0,0.3,0.7",
        help="Grid of reranker.hard_negative_ratio.",
    )
    parser.add_argument(
        "--hard-negative-skip-top-frac",
        type=float,
        default=0.1,
        help="Fixed across the sweep; set to 0 to disable skip.",
    )
    parser.add_argument(
        "--negative-sampling",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--train-sample-n",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
    )
    args = parser.parse_args()

    import numpy as np

    from core.config import load_config
    from recommender.src.data.preprocess import load_processed
    from recommender.src.evaluation.evaluate import (
        evaluate_pretrained_als,
        leave_one_out_split,
    )
    from recommender.src.features.build_matrix import (
        build_user_item_matrix,
        get_user_item_mappers,
    )
    from recommender.src.models.als import train_als
    from recommender.src.models.reranker import (
        ReRankerConfig,
        build_reranker_training_frame,
        train_reranker,
    )
    from recommender.src.retrieval.candidates import build_retrieval_metadata

    project_cfg: dict = {}
    if args.config is not None:
        project_cfg = load_config(args.config) or {}
        if project_cfg.get("seed") is not None:
            args.random_state = int(project_cfg["seed"])

    als_defaults = (project_cfg.get("als") or {}) if project_cfg else {}
    als_cfg = {
        "factors": int(als_defaults.get("factors", 64)),
        "regularization": float(als_defaults.get("regularization", 0.01)),
        "iterations": int(als_defaults.get("iterations", 15)),
        "alpha": float(als_defaults.get("alpha", 10.0)),
    }

    model_types = [s.strip() for s in args.model_types.split(",") if s.strip()]
    for mt in model_types:
        if mt not in {"linear", "pointwise"}:
            raise SystemExit(f"Unsupported model_type in grid: {mt}")
    candidate_top_ns = _parse_int_list(args.candidate_top_n)
    hnr_values = _parse_float_list(args.hard_negative_ratio)

    interactions, albums = load_processed(Path(args.processed_dir))
    if interactions.empty:
        raise SystemExit("No interactions found in processed_dir.")
    if albums is None or albums.empty:
        raise SystemExit(
            "albums.parquet is required for reranker features "
            "(RetrievalMetadata build)."
        )

    if args.sample_n <= 0 or args.sample_n >= len(interactions):
        sample = interactions.copy()
    else:
        sample = interactions.sample(
            n=args.sample_n, random_state=args.random_state
        )
    user_counts = sample["user_id"].value_counts()
    keep_users = user_counts[
        user_counts >= args.min_interactions_per_user
    ].index.astype(str)
    sample = sample[sample["user_id"].astype(str).isin(keep_users)].copy()
    sample["user_id"] = sample["user_id"].astype(str)
    sample["album_id"] = sample["album_id"].astype(str)

    train_int, test_int = leave_one_out_split(
        sample, random_state=args.random_state
    )

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

    grid = list(product(model_types, candidate_top_ns, hnr_values))

    print("=== Reranker sweep ===")
    print(f"sample rows: {len(sample)} users: {sample['user_id'].nunique()}")
    print(f"items (matrix cols): {len(item_ids)}")
    print(f"als_cfg (fixed): {als_cfg}")
    print(f"grid size: {len(grid)}")

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

    t0 = time.time()
    base_out = evaluate_pretrained_als(
        model,
        matrix,
        user_id2idx,
        item_id2idx,
        item_ids,
        train_int,
        test_int,
        args.k,
    )
    base_s = time.time() - t0
    print(f"ALS-only eval time: {base_s:.1f}s")
    print(
        f"ALS-only: ndcg@{args.k}={base_out.get(f'ndcg@{args.k}', 0.0):.5f} "
        f"pop_head={base_out.get(f'ndcg@{args.k}_pop_head', 0.0):.5f} "
        f"pop_tail={base_out.get(f'ndcg@{args.k}_pop_tail', 0.0):.5f}"
    )

    t0 = time.time()
    meta = build_retrieval_metadata(albums, train_int)
    print(f"RetrievalMetadata build time: {time.time() - t0:.1f}s")
    if not meta.valid_album_ids:
        raise SystemExit(
            "RetrievalMetadata is empty; reranker cannot be evaluated."
        )

    results: list[dict[str, object]] = []
    als_only_ndcg = float(base_out.get(f"ndcg@{args.k}", 0.0))
    als_only_pop_head = float(base_out.get(f"ndcg@{args.k}_pop_head", 0.0))
    als_only_pop_tail = float(base_out.get(f"ndcg@{args.k}_pop_tail", 0.0))

    for run_idx, (mt, cn, hnr) in enumerate(grid, start=1):
        rr_cfg = ReRankerConfig(
            enabled=True,
            model_type=mt,
            candidate_top_n=int(cn),
            train_sample_n=int(args.train_sample_n),
            negative_sampling=int(args.negative_sampling),
            class_weight="balanced",
            hard_negative_ratio=float(hnr),
            hard_negative_skip_top_frac=float(
                args.hard_negative_skip_top_frac
            ),
            random_state=args.random_state,
        )
        print(
            f"\n[{run_idx}/{len(grid)}] model={mt} top_n={cn} hnr={hnr} "
            f"(skip_top={args.hard_negative_skip_top_frac})"
        )
        t0 = time.time()
        rr_df, rr_stats = build_reranker_training_frame(
            model=model,
            item_ids=item_ids,
            user_id2idx=user_id2idx,
            item_id2idx=item_id2idx,
            train_interactions=train_int,
            test_interactions=test_int,
            meta=meta,
            rr_cfg=rr_cfg,
        )
        frame_s = time.time() - t0
        bundle = train_reranker(rr_df, rr_cfg)
        train_s = time.time() - t0 - frame_s
        row: dict[str, object] = {
            "run": run_idx,
            "rr_cfg": asdict(rr_cfg),
            "rr_stats": {k: float(v) for k, v in rr_stats.items()},
            "frame_build_s": float(frame_s),
            "reranker_train_s": float(train_s),
        }
        if bundle is None:
            row["metrics"] = {}
            row["skipped"] = "train_reranker returned None"
            print("  -> reranker training returned None; skipping eval")
            results.append(row)
            continue

        t0 = time.time()
        reranked = evaluate_pretrained_als(
            model,
            matrix,
            user_id2idx,
            item_id2idx,
            item_ids,
            train_int,
            test_int,
            args.k,
            meta=meta,
            reranker_bundle=bundle,
        )
        eval_s = time.time() - t0
        row["rerank_eval_s"] = float(eval_s)

        metrics = {
            f"ndcg@{args.k}": float(reranked.get(f"ndcg@{args.k}", 0.0)),
            f"map@{args.k}": float(reranked.get(f"map@{args.k}", 0.0)),
            f"recall@{args.k}": float(reranked.get(f"recall@{args.k}", 0.0)),
            f"ndcg@{args.k}_pop_head": float(
                reranked.get(f"ndcg@{args.k}_pop_head", 0.0)
            ),
            f"ndcg@{args.k}_pop_tail": float(
                reranked.get(f"ndcg@{args.k}_pop_tail", 0.0)
            ),
            f"als_only_ndcg@{args.k}": als_only_ndcg,
            f"als_only_ndcg@{args.k}_pop_head": als_only_pop_head,
            f"als_only_ndcg@{args.k}_pop_tail": als_only_pop_tail,
            f"als_only_map@{args.k}": float(
                base_out.get(f"map@{args.k}", 0.0)
            ),
            f"als_only_recall@{args.k}": float(
                base_out.get(f"recall@{args.k}", 0.0)
            ),
        }
        row["metrics"] = metrics
        results.append(row)
        print(
            f"  ndcg@{args.k}={metrics[f'ndcg@{args.k}']:.5f} "
            f"(als={als_only_ndcg:.5f}) "
            f"pop_head={metrics[f'ndcg@{args.k}_pop_head']:.5f} "
            f"(als={als_only_pop_head:.5f}) "
            f"pop_tail={metrics[f'ndcg@{args.k}_pop_tail']:.5f} "
            f"(als={als_only_pop_tail:.5f})"
        )

    winner: dict[str, object] | None = None
    winner_head_delta = float("inf")
    for row in results:
        m = row.get("metrics") or {}
        if not isinstance(m, dict) or f"ndcg@{args.k}" not in m:
            continue
        n_overall = m[f"ndcg@{args.k}"]
        n_tail = m[f"ndcg@{args.k}_pop_tail"]
        n_head = m[f"ndcg@{args.k}_pop_head"]
        if n_overall <= als_only_ndcg:
            continue
        if n_tail <= als_only_pop_tail:
            continue
        head_delta = abs(als_only_pop_head - n_head)
        if head_delta < winner_head_delta:
            winner_head_delta = head_delta
            winner = row

    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep_reranker.json"
    payload = {
        "als_cfg": als_cfg,
        "als_only": {
            f"ndcg@{args.k}": als_only_ndcg,
            f"ndcg@{args.k}_pop_head": als_only_pop_head,
            f"ndcg@{args.k}_pop_tail": als_only_pop_tail,
            f"map@{args.k}": float(base_out.get(f"map@{args.k}", 0.0)),
            f"recall@{args.k}": float(base_out.get(f"recall@{args.k}", 0.0)),
        },
        "grid_size": len(grid),
        "results": results,
        "winner": winner,
        "winner_head_delta": (
            winner_head_delta if winner is not None else None
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to: {out_path}")
    if winner is None:
        print(
            "No config satisfied (ndcg@k > als_only) AND "
            "(pop_tail > als_only_pop_tail). "
            "Keep reranker.enabled: false."
        )
    else:
        print(
            "Winner: "
            f"{winner['rr_cfg']}"  # type: ignore[index]
            f"\n  head_delta={winner_head_delta:.5f} "
            f"(lower is better; 0 = no pop_head regression)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
