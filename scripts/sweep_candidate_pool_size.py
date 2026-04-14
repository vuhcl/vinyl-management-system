#!/usr/bin/env python3
"""
Sweep two-stage ``max_candidates`` with a *single* ALS fit (fast comparison).

Trains ALS once on the train split, then re-ranks with different pool caps.
Also runs a full-catalog baseline (no candidate restriction).

Example:

  python scripts/sweep_candidate_pool_size.py \\
    --processed-dir data/processed \\
    --sample-n 200000 \\
    --max-candidates-list 500,1000,2000,5000,10000,20000

Outputs a table and optional JSON (``--artifacts-dir``).
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep max_candidates for two-stage ALS (single model fit).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=200_000,
        help="Interaction rows to use (0 = full dataset)",
    )
    parser.add_argument(
        "--min-interactions-per-user",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--max-candidates-list",
        type=str,
        default="500,1000,2000,5000,10000,20000,40000",
        help="Comma-separated max_candidates values to try",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="If set, write sweep_candidate_pool.json here",
    )
    args = parser.parse_args()

    import numpy as np

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
    from recommender.src.retrieval.candidates import (
        build_retrieval_metadata,
        retrieval_config_from_dict,
    )

    sizes = [int(x.strip()) for x in args.max_candidates_list.split(",") if x.strip()]

    interactions, albums = load_processed(Path(args.processed_dir))
    if interactions.empty or albums.empty:
        raise SystemExit("Need non-empty interactions and albums.parquet.")

    if args.sample_n <= 0 or args.sample_n >= len(interactions):
        sample = interactions.copy()
    else:
        sample = interactions.sample(
            n=min(args.sample_n, len(interactions)),
            random_state=args.random_state,
        )
    vc = sample["user_id"].value_counts()
    keep = vc[vc >= args.min_interactions_per_user].index.astype(str)
    sample = sample[sample["user_id"].astype(str).isin(keep)].copy()
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

    als_cfg = {
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "alpha": args.alpha,
    }
    print("Fitting ALS once (train rows=%d)..." % len(train_int))
    t0 = time.perf_counter()
    model = train_als(
        matrix,
        factors=als_cfg["factors"],
        regularization=als_cfg["regularization"],
        iterations=als_cfg["iterations"],
        alpha=als_cfg["alpha"],
        random_state=args.random_state,
    )
    print("  train time: %.1fs" % (time.perf_counter() - t0))

    base_retrieval = {
        "enabled": True,
        "min_avg_rating": 0.0,
        "min_train_count": 1,
        "use_genre_expansion": True,
        "use_same_artist_expansion": True,
        "fallback_to_full_catalog": True,
        "use_year_quantile_filter": True,
        "year_quantile_low": 0.1,
        "year_quantile_high": 0.9,
        "min_distinct_users": 1,
        "min_rating_rows": 0,
    }
    meta = build_retrieval_metadata(albums, train_int)
    if not meta.valid_album_ids:
        raise SystemExit("Retrieval metadata empty; check albums.parquet.")

    results: list[dict[str, object]] = []

    print("\n=== Full catalog (no candidate cap) ===")
    t1 = time.perf_counter()
    full_m = evaluate_pretrained_als(
        model,
        matrix,
        user_id2idx,
        item_id2idx,
        item_ids,
        train_int,
        test_int,
        args.k,
        meta=None,
        retrieval_cfg=None,
    )
    full_m["label"] = "full_catalog"
    full_m["max_candidates"] = None
    full_m["eval_s"] = time.perf_counter() - t1
    results.append(full_m)
    print(full_m)

    hdr = (
        f"{'max_k':>8} {'ndcg@k':>10} {'map@k':>10} {'recall@k':>10} "
        f"{'hit_rate':>10} {'eval_s':>8}"
    )
    print("\n=== Two-stage sweep (same ALS) ===")
    print(hdr)
    kstr = f"@{args.k}"
    for mc in sizes:
        rcfg = retrieval_config_from_dict({**base_retrieval, "max_candidates": mc})
        t2 = time.perf_counter()
        m = evaluate_pretrained_als(
            model,
            matrix,
            user_id2idx,
            item_id2idx,
            item_ids,
            train_int,
            test_int,
            args.k,
            meta=meta,
            retrieval_cfg=rcfg,
        )
        ev_s = time.perf_counter() - t2
        row = {
            "label": "two_stage",
            "max_candidates": mc,
            "eval_s": ev_s,
            **m,
        }
        results.append(row)
        ndcg = m.get(f"ndcg{kstr}", 0.0)
        mp = m.get(f"map{kstr}", 0.0)
        rec = m.get(f"recall{kstr}", 0.0)
        hit = m.get("candidate_relevant_hit_rate", float("nan"))
        print(
            f"{mc:>8} {ndcg:>10.6f} {mp:>10.6f} {rec:>10.6f} "
            f"{hit:>10.4f} {ev_s:>8.1f}"
        )

    best_ts = max(
        (r for r in results if r.get("max_candidates") is not None),
        key=lambda r: float(r.get(f"ndcg@{args.k}", 0.0)),
        default=None,
    )
    if best_ts:
        print(
            "\nBest max_candidates by NDCG@{}: {} (ndcg={:.6f})".format(
                args.k,
                best_ts["max_candidates"],
                float(best_ts.get(f"ndcg@{args.k}", 0.0)),
            )
        )

    if args.artifacts_dir is not None:
        out_dir = Path(args.artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sweep_candidate_pool.json"
        payload = {
            "k": args.k,
            "als": als_cfg,
            "sample_rows": len(sample),
            "train_rows": len(train_int),
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
