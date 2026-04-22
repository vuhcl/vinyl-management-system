"""
End-to-end pipeline: ingest → preprocess → features → train → evaluate → recommend.
Run from project root: python -m recommender.pipeline (or PYTHONPATH=. python recommender/pipeline.py)
"""
from pathlib import Path
import json
import pickle
import yaml

import numpy as np
import pandas as pd

from core.config import get_project_root, load_config as load_project_config

from recommender.src.data.ingest import ingest_all
from recommender.src.data.preprocess import preprocess, save_processed, load_processed
from recommender.src.features.build_matrix import build_user_item_matrix, get_user_item_mappers
from recommender.src.features.content_features import prepare_album_features
from recommender.src.models.als import predict_als, train_als
from recommender.src.models.reranker import (
    build_reranker_training_frame,
    load_reranker_bundle,
    rerank_candidates_for_user,
    reranker_config_from_dict,
    save_reranker_bundle,
    train_reranker,
)
from recommender.src.retrieval.candidates import (
    RetrievalMetadata,
    build_retrieval_metadata,
)
from recommender.src.models.content_model import build_content_similarity, content_scores_vector
from recommender.src.models.hybrid import rank_hybrid
from recommender.src.evaluation.evaluate import leave_one_out_split, run_evaluation


def run_pipeline(
    config_path: Path,
    data_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    skip_ingest: bool = False,
    log_mlflow: bool = True,
):
    config = load_project_config(config_path)
    seed = config.get("seed", 42)
    np.random.seed(seed)
    weights = config.get("interaction_weights", {})
    als_cfg = config.get("als", {})
    hybrid_cfg = config.get("hybrid", {})
    eval_cfg = config.get("evaluation", {})
    reranker_cfg = config.get("reranker") or {}
    k = eval_cfg.get("k", 10)

    data_dir = Path(data_dir)
    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not skip_ingest:
        discogs_cfg = config.get("discogs", {})
        aoty_cfg = config.get("aoty_scraped", {})
        project_root = get_project_root()
        aoty_dir = aoty_cfg.get("dir")
        if aoty_dir is not None:
            aoty_dir = Path(aoty_dir)
            if not aoty_dir.is_absolute():
                aoty_dir = project_root / aoty_dir
        release_map_path = discogs_cfg.get("release_to_aoty_map_path")
        if release_map_path:
            rp = Path(release_map_path).expanduser()
            if not rp.is_absolute():
                release_map_path = str(project_root / rp)
            else:
                release_map_path = str(rp)
        raw = ingest_all(
            data_dir,
            discogs={
                "use_api": discogs_cfg.get("use_api", False),
                "usernames": discogs_cfg.get("usernames"),
                "token": discogs_cfg.get("token"),
                "release_to_aoty_map_path": release_map_path,
                "skip_live_discogs_aoty_mapping": discogs_cfg.get(
                    "skip_live_discogs_aoty_mapping", False
                ),
            } if discogs_cfg else None,
            aoty_scraped={
                "dir": aoty_dir,
                "ratings_file": aoty_cfg.get("ratings_file", "ratings.csv"),
                "albums_file": aoty_cfg.get("albums_file", "albums.csv"),
            } if aoty_cfg else None,
        )
        interactions, albums = preprocess(raw, weights)
        save_processed(interactions, albums, processed_dir)
    else:
        interactions, albums = load_processed(processed_dir)

    if interactions.empty:
        raise ValueError("No interactions after preprocess. Add data in data/raw/.")

    # Feature matrix and mappers
    matrix, user_ids, item_ids = build_user_item_matrix(interactions, weight_col="strength")
    user_id2idx, item_id2idx, idx2user_id, idx2item_id = get_user_item_mappers(user_ids, item_ids)

    # Content features (optional)
    content_sim = None
    album_features = None
    if not albums.empty and "album_id" in albums.columns:
        content_cfg = config.get("content", {})
        max_items = content_cfg.get("max_items_for_similarity", 5000)
        if len(item_ids) > max_items:
            print(
                f"Skipping content similarity: {len(item_ids)} items > {max_items}"
            )
        else:
            album_features, _ = prepare_album_features(
                albums,
                top_k_genres=content_cfg.get("genre_top_k", 20),
            )
            album_features = (
                album_features.reindex(index=item_ids.astype(str)).fillna(0)
            )
            content_sim = build_content_similarity(album_features)

    # Train/test split and evaluate
    train_int, test_int = leave_one_out_split(interactions, random_state=seed)
    metrics = run_evaluation(
        train_int,
        test_int,
        als_cfg,
        k=k,
        random_state=seed,
        albums=albums if not albums.empty else None,
        reranker=reranker_cfg if reranker_cfg else None,
    )
    print("Evaluation metrics:", metrics)

    # Train final ALS on full data
    model = train_als(
        matrix,
        factors=als_cfg.get("factors", 64),
        regularization=als_cfg.get("regularization", 0.01),
        iterations=als_cfg.get("iterations", 15),
        alpha=als_cfg.get("alpha", 40.0),
        random_state=seed,
    )

    # Save artifacts for serving
    from scipy.sparse import save_npz
    with open(artifacts_dir / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)
    save_npz(artifacts_dir / "user_item_matrix.npz", matrix)
    with open(artifacts_dir / "mappers.pkl", "wb") as f:
        pickle.dump({
            "user_id2idx": user_id2idx,
            "item_id2idx": item_id2idx,
            "idx2user_id": idx2user_id,
            "idx2item_id": idx2item_id,
            "user_ids": user_ids,
            "item_ids": item_ids,
        }, f)
    if content_sim is not None:
        np.save(artifacts_dir / "content_sim.npy", content_sim)
    with open(artifacts_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f)
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Build and save retrieval metadata for reranker serving
    retrieval_meta: RetrievalMetadata | None = None
    reranker_bundle = None
    rr_cfg_obj = reranker_config_from_dict(reranker_cfg)

    if not albums.empty:
        retrieval_meta = build_retrieval_metadata(albums, interactions)
        with open(artifacts_dir / "retrieval_serving.pkl", "wb") as f:
            pickle.dump({"meta": retrieval_meta}, f)

        if rr_cfg_obj.enabled and retrieval_meta.valid_album_ids:
            rr_df, rr_stats = build_reranker_training_frame(
                model=model,
                item_ids=item_ids,
                user_id2idx=user_id2idx,
                item_id2idx=item_id2idx,
                train_interactions=train_int,
                test_interactions=test_int,
                meta=retrieval_meta,
                rr_cfg=rr_cfg_obj,
            )
            reranker_bundle = train_reranker(rr_df, rr_cfg_obj)
            if reranker_bundle is not None:
                save_reranker_bundle(reranker_bundle, artifacts_dir / "reranker.pkl")
                with open(artifacts_dir / "reranker_meta.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model_type": reranker_bundle.model_type,
                            "candidate_top_n": reranker_bundle.candidate_top_n,
                            "train_rows": reranker_bundle.train_rows,
                            "positive_rate": reranker_bundle.positive_rate,
                            **rr_stats,
                        },
                        f,
                        indent=2,
                    )
                with open(artifacts_dir / "reranker_champion.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "enabled": True,
                            "model_type": reranker_bundle.model_type,
                            "artifact": "reranker.pkl",
                            "primary_metric": f"ndcg@{k}",
                            "score": float(metrics.get(f"ndcg@{k}", 0.0)),
                        },
                        f,
                        indent=2,
                    )
    if log_mlflow:
        try:
            import mlflow
            import os
            uri = os.environ.get("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("recommender-als")
            with mlflow.start_run(run_name="final_train"):
                mlflow.log_params(als_cfg)
                mlflow.log_metrics(metrics)
                if reranker_bundle is not None:
                    mlflow.log_params(
                        {
                            "reranker.enabled": True,
                            "reranker.model_type": reranker_bundle.model_type,
                            "reranker.candidate_top_n": reranker_bundle.candidate_top_n,
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "reranker_train_rows": float(reranker_bundle.train_rows),
                            "reranker_positive_rate": float(reranker_bundle.positive_rate),
                        }
                    )
        except Exception:
            pass

    return {
        "model": model,
        "matrix": matrix,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "user_id2idx": user_id2idx,
        "item_id2idx": item_id2idx,
        "idx2user_id": idx2user_id,
        "idx2item_id": idx2item_id,
        "content_sim": content_sim,
        "metrics": metrics,
        "config": config,
        "retrieval_meta": retrieval_meta,
        "reranker_bundle": reranker_bundle,
    }


def load_pipeline_artifacts(artifacts_dir: Path) -> dict | None:
    """
    Load saved artifacts (model, mappers, matrix, content_sim) for serving.
    Returns None if any required file is missing.
    """
    artifacts_dir = Path(artifacts_dir)
    if not (artifacts_dir / "als_model.pkl").exists() or not (artifacts_dir / "mappers.pkl").exists():
        return None
    with open(artifacts_dir / "als_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(artifacts_dir / "mappers.pkl", "rb") as f:
        mappers = pickle.load(f)
    from scipy.sparse import load_npz
    matrix_path = artifacts_dir / "user_item_matrix.npz"
    if not matrix_path.exists():
        return None
    matrix = load_npz(matrix_path)
    content_sim = None
    if (artifacts_dir / "content_sim.npy").exists():
        content_sim = np.load(artifacts_dir / "content_sim.npy", allow_pickle=True)
        if hasattr(content_sim, "item"):
            content_sim = content_sim.item()
    retrieval_meta = None
    rs_path = artifacts_dir / "retrieval_serving.pkl"
    if rs_path.exists():
        try:
            with open(rs_path, "rb") as f:
                rs = pickle.load(f)
            retrieval_meta = rs.get("meta")
        except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
            print(
                "retrieval_serving.pkl is from a pre-ALS-full-catalog run; "
                "re-run recommender.pipeline to regenerate."
            )
    out = {
        "model": model,
        "matrix": matrix,
        "user_ids": mappers["user_ids"],
        "item_ids": mappers["item_ids"],
        "user_id2idx": mappers["user_id2idx"],
        "item_id2idx": mappers["item_id2idx"],
        "idx2user_id": mappers["idx2user_id"],
        "idx2item_id": mappers["idx2item_id"],
        "content_sim": content_sim,
    }
    if retrieval_meta is not None:
        out["retrieval_meta"] = retrieval_meta
    rr_path = artifacts_dir / "reranker.pkl"
    rr_bundle = load_reranker_bundle(rr_path)
    if rr_bundle is not None:
        out["reranker_bundle"] = rr_bundle
    return out


def recommend(
    user_id: str,
    pipeline_artifacts: dict,
    top_k: int = 10,
    exclude_owned: bool = True,
    alpha: float = 0.7,
) -> dict:
    """
    Return recommendations for user_id.

    Flow: hybrid content blending (if ``content_sim`` and ``alpha`` < 1)
    → reranker over full-catalog ALS top-N (if reranker bundle + retrieval meta)
    → full-catalog ``predict_als``.

    Returns:
        {"user_id": "...", "recommendations": [{"album_id": "...", "score": float, "rank": int}, ...]}
    """
    user_id2idx = pipeline_artifacts["user_id2idx"]
    item_id2idx = pipeline_artifacts["item_id2idx"]
    idx2item_id = pipeline_artifacts["idx2item_id"]
    model = pipeline_artifacts["model"]
    matrix = pipeline_artifacts["matrix"]
    item_ids = pipeline_artifacts["item_ids"]
    content_sim = pipeline_artifacts.get("content_sim")
    retrieval_meta = pipeline_artifacts.get("retrieval_meta")
    reranker_bundle = pipeline_artifacts.get("reranker_bundle")
    n_items = len(item_ids)

    if user_id not in user_id2idx:
        return {"user_id": user_id, "recommendations": []}

    user_idx = user_id2idx[user_id]
    owned = set(matrix[user_idx].indices) if exclude_owned else set()
    exclude_idxs = np.array(list(owned), dtype=int)

    hybrid_on = content_sim is not None and alpha < 1.0
    if hybrid_on:
        cf_scores = np.zeros(n_items, dtype=np.float64)
        for i in range(n_items):
            if i in owned:
                continue
            cf_scores[i] = float(
                np.dot(model.user_factors[user_idx], model.item_factors[i])
            )
        user_album_ids = [idx2item_id[i] for i in matrix[user_idx].indices]
        content_scores = content_scores_vector(
            np.array(user_album_ids), item_id2idx, n_items, content_sim, exclude_idxs
        )
        exclude_mask = np.zeros(n_items, dtype=bool)
        exclude_mask[exclude_idxs] = True
        rank_idx, scores = rank_hybrid(
            cf_scores, content_scores, alpha, exclude_mask, top_k
        )
    elif reranker_bundle is not None and retrieval_meta is not None:
        train_albums = {str(idx2item_id[int(i)]) for i in matrix[user_idx].indices}
        cand = np.arange(n_items, dtype=np.int64)
        rank_idx, scores = rerank_candidates_for_user(
            bundle=reranker_bundle,
            model=model,
            user_idx=user_idx,
            train_albums=train_albums,
            candidate_item_idxs=cand,
            exclude_idxs=exclude_idxs,
            item_ids=item_ids,
            meta=retrieval_meta,
            top_k=top_k,
        )
    else:
        rank_idx, scores = predict_als(
            model, user_idx, matrix, item_ids, exclude_idxs, top_k=top_k
        )

    recs = [
        {"album_id": idx2item_id[int(i)], "score": float(s), "rank": r}
        for r, (i, s) in enumerate(zip(rank_idx, scores), start=1)
    ]
    return {"user_id": user_id, "recommendations": recs}


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from shared.project_env import load_project_dotenv

    load_project_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="recommender/configs/base.yaml",
        help="YAML (may use inherits: configs/base.yaml)",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true", dest="no_mlflow")
    args = parser.parse_args()
    run_pipeline(
        Path(args.config),
        Path(args.data_dir),
        Path(args.processed_dir),
        Path(args.artifacts_dir),
        skip_ingest=args.skip_ingest,
        log_mlflow=not args.no_mlflow,
    )
