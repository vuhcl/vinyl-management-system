"""
Second-stage re-ranker for ALS top-N candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ..retrieval.candidates import RetrievalMetadata


@dataclass
class ReRankerConfig:
    enabled: bool = False
    model_type: str = "linear"  # linear | pointwise
    candidate_top_n: int = 500
    train_sample_n: int = 250000
    negative_sampling: int = 200
    class_weight: str | None = "balanced"
    hard_negative_ratio: float = 0.7
    user_chunk_size: int = 10000
    random_state: int = 42


@dataclass
class ReRankerBundle:
    model_type: str
    model: object
    feature_names: list[str]
    candidate_top_n: int
    train_rows: int
    positive_rate: float


def reranker_config_from_dict(d: dict | None) -> ReRankerConfig:
    if not d:
        return ReRankerConfig(enabled=False)
    return ReRankerConfig(
        enabled=bool(d.get("enabled", False)),
        model_type=str(d.get("model_type", "linear")),
        candidate_top_n=int(d.get("candidate_top_n", 500)),
        train_sample_n=int(d.get("train_sample_n", 250000)),
        negative_sampling=int(d.get("negative_sampling", 200)),
        class_weight=d.get("class_weight", "balanced"),
        hard_negative_ratio=float(d.get("hard_negative_ratio", 0.7)),
        user_chunk_size=int(d.get("user_chunk_size", 10000)),
        random_state=int(d.get("random_state", 42)),
    )


def _genre_jaccard(user_genres: set[str], cand_genres: set[str]) -> float:
    if not user_genres and not cand_genres:
        return 0.0
    union = user_genres | cand_genres
    if not union:
        return 0.0
    return float(len(user_genres & cand_genres) / len(union))


def topn_als_from_candidates(
    model,
    user_idx: int,
    candidate_item_idxs: np.ndarray,
    exclude_idxs: np.ndarray,
    *,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-N candidate indices by ALS dot product."""
    if candidate_item_idxs.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    inds = np.asarray(candidate_item_idxs, dtype=np.int64)
    exclude_set = set(int(x) for x in np.asarray(exclude_idxs).ravel())
    uf = np.asarray(model.user_factors[user_idx], dtype=np.float64)
    IF = np.asarray(model.item_factors[inds], dtype=np.float64)
    scores = IF @ uf
    order = np.argsort(-scores)
    picked_idx: list[int] = []
    picked_scores: list[float] = []
    for j in order:
        idx = int(inds[j])
        if idx in exclude_set:
            continue
        picked_idx.append(idx)
        picked_scores.append(float(scores[j]))
        if len(picked_idx) >= top_n:
            break
    return np.asarray(picked_idx, dtype=np.int64), np.asarray(
        picked_scores, dtype=np.float64
    )


def _feature_rows(
    *,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    item_ids: np.ndarray,
    train_albums: set[str],
    meta: RetrievalMetadata,
    item_train_counts: dict[str, int],
) -> list[dict[str, float]]:
    user_genres: set[str] = set()
    for a in train_albums:
        user_genres |= meta.album_genres.get(str(a), set())
    user_artist_set = {
        meta.album_id_to_artist.get(str(a), "")
        for a in train_albums
        if meta.album_id_to_artist.get(str(a), "")
    }
    years = [
        meta.album_year.get(str(a), 0)
        for a in train_albums
        if meta.album_year.get(str(a), 0) > 0
    ]
    user_year = float(np.mean(years)) if years else 0.0

    rows: list[dict[str, float]] = []
    for rank0, (idx, als_s) in enumerate(zip(top_idx, top_scores)):
        aid = str(item_ids[int(idx)])
        cg = meta.album_genres.get(aid, set())
        c_artist = meta.album_id_to_artist.get(aid, "")
        c_year = float(meta.album_year.get(aid, 0))
        rows.append(
            {
                "als_score": float(als_s),
                "als_rank_inv": float(1.0 / (1.0 + rank0)),
                "genre_jaccard": _genre_jaccard(user_genres, cg),
                "artist_match": (
                    1.0 if c_artist and c_artist in user_artist_set else 0.0
                ),
                "item_popularity": float(item_train_counts.get(aid, 0)),
                "year_distance": (
                    abs(c_year - user_year)
                    if user_year > 0 and c_year > 0
                    else 0.0
                ),
                "item_avg_rating": float(meta.album_avg_rating.get(aid, 0.0)),
                "item_priority": float(meta.album_priority.get(aid, 0.0)),
                "item_distinct_users": float(
                    meta.album_distinct_users.get(aid, 0)
                ),
            }
        )
    return rows


def build_reranker_training_frame(
    *,
    model,
    item_ids: np.ndarray,
    user_id2idx: dict[str, int],
    item_id2idx: dict[str, int],
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    meta: RetrievalMetadata,
    rr_cfg: ReRankerConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build sparse pointwise training frame from ALS top-N candidates over the full item matrix."""
    rng = np.random.default_rng(rr_cfg.random_state)
    test_by_user = (
        test_interactions.groupby("user_id")["album_id"].apply(set).to_dict()
    )
    _ti = train_interactions.assign(
        user_id=train_interactions["user_id"].astype(str),
        album_id=train_interactions["album_id"].astype(str),
    )
    train_by_user = _ti.groupby("user_id")["album_id"].apply(set).to_dict()
    item_train_counts: dict[str, int] = {
        str(a): int(v)
        for a, v in train_interactions.groupby("album_id", sort=False)
        .size()
        .items()
    }

    users = [u for u in test_by_user.keys() if u in user_id2idx]
    if rr_cfg.train_sample_n > 0 and len(users) > rr_cfg.train_sample_n:
        users = rng.choice(
            users, size=rr_cfg.train_sample_n, replace=False
        ).tolist()

    # Stage-1 is always full-catalog ALS: consider all item indices.
    cand = np.arange(len(item_ids), dtype=np.int64)

    rows: list[dict[str, float | int | str]] = []
    users_no_pos = 0
    for uid in users:
        train_items = train_by_user.get(uid, set())
        user_idx = user_id2idx[uid]
        relevant_idx = {
            item_id2idx[a]
            for a in test_by_user.get(uid, set())
            if a in item_id2idx
        }
        if not relevant_idx:
            continue
        exclude_idxs = np.array(
            [item_id2idx[a] for a in train_items if a in item_id2idx],
            dtype=np.int64,
        )
        top_idx, top_scores = topn_als_from_candidates(
            model,
            user_idx,
            cand,
            exclude_idxs,
            top_n=rr_cfg.candidate_top_n,
        )
        if top_idx.size == 0:
            users_no_pos += 1
            continue

        feat_rows = _feature_rows(
            top_idx=top_idx,
            top_scores=top_scores,
            item_ids=item_ids,
            train_albums=train_items,
            meta=meta,
            item_train_counts=item_train_counts,
        )
        u_rows: list[dict[str, float | int | str]] = []
        for i, fr in enumerate(feat_rows):
            idx = int(top_idx[i])
            label = 1 if idx in relevant_idx else 0
            u_rows.append(
                {"user_id": uid, "item_idx": idx, "label": label, **fr}
            )
        pos_rows = [r for r in u_rows if int(r["label"]) == 1]
        neg_rows = [r for r in u_rows if int(r["label"]) == 0]
        if not pos_rows:
            users_no_pos += 1
        keep_neg = min(len(neg_rows), rr_cfg.negative_sampling)
        if keep_neg > 0:
            hard_n = int(round(keep_neg * rr_cfg.hard_negative_ratio))
            hard = neg_rows[:hard_n]
            tail = neg_rows[hard_n:]
            random_keep = []
            if keep_neg > hard_n and tail:
                kk = min(keep_neg - hard_n, len(tail))
                ix = rng.choice(len(tail), size=kk, replace=False)
                random_keep = [
                    tail[int(j)] for j in np.asarray(ix).ravel().tolist()
                ]
            rows.extend(pos_rows + hard + random_keep)
        else:
            rows.extend(pos_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "rr_train_rows": 0.0,
            "rr_positive_rate": 0.0,
            "rr_users_sampled": float(len(users)),
            "rr_users_no_positive_in_pool": float(users_no_pos),
        }
    stats = {
        "rr_train_rows": float(len(df)),
        "rr_positive_rate": float(df["label"].mean()),
        "rr_users_sampled": float(len(users)),
        "rr_users_no_positive_in_pool": float(users_no_pos),
    }
    return df, stats


def train_reranker(
    train_df: pd.DataFrame,
    rr_cfg: ReRankerConfig,
) -> ReRankerBundle | None:
    if train_df.empty:
        return None
    feature_names = [
        "als_score",
        "als_rank_inv",
        "genre_jaccard",
        "artist_match",
        "item_popularity",
        "year_distance",
        "item_avg_rating",
        "item_priority",
        "item_distinct_users",
    ]
    X = train_df[feature_names].astype(float).values
    y = train_df["label"].astype(int).values
    if np.unique(y).shape[0] < 2:
        return None

    if rr_cfg.model_type == "linear":
        model = LogisticRegression(
            max_iter=250,
            class_weight=rr_cfg.class_weight,
            random_state=rr_cfg.random_state,
        )
        model.fit(X, y)
    elif rr_cfg.model_type == "pointwise":
        pos = max(1, int(y.sum()))
        neg = max(1, int((1 - y).sum()))
        scale_pos = float(neg / pos)
        sw = np.where(y == 1, scale_pos, 1.0).astype(np.float64)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=rr_cfg.random_state,
        )
        model.fit(X, y, sample_weight=sw)
    else:
        raise ValueError(
            f"Unsupported reranker model_type: {rr_cfg.model_type}"
        )

    return ReRankerBundle(
        model_type=rr_cfg.model_type,
        model=model,
        feature_names=feature_names,
        candidate_top_n=rr_cfg.candidate_top_n,
        train_rows=len(train_df),
        positive_rate=float(train_df["label"].mean()),
    )


def predict_reranker_scores(
    bundle: ReRankerBundle, feat_df: pd.DataFrame
) -> np.ndarray:
    X = feat_df[bundle.feature_names].astype(float).values
    m = bundle.model
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=np.float64)
    if hasattr(m, "decision_function"):
        return np.asarray(m.decision_function(X), dtype=np.float64)
    return np.asarray(m.predict(X), dtype=np.float64)


def rerank_candidates_for_user(
    *,
    bundle: ReRankerBundle,
    model,
    user_idx: int,
    train_albums: set[str],
    candidate_item_idxs: np.ndarray,
    exclude_idxs: np.ndarray,
    item_ids: np.ndarray,
    meta: RetrievalMetadata,
    item_train_counts: dict[str, int],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_idx, top_scores = topn_als_from_candidates(
        model,
        user_idx,
        candidate_item_idxs,
        exclude_idxs,
        top_n=bundle.candidate_top_n,
    )
    if top_idx.size == 0:
        return top_idx, np.array([], dtype=np.float64)
    rows = _feature_rows(
        top_idx=top_idx,
        top_scores=top_scores,
        item_ids=item_ids,
        train_albums=train_albums,
        meta=meta,
        item_train_counts=item_train_counts,
    )
    feat_df = pd.DataFrame(rows)
    rr_scores = predict_reranker_scores(bundle, feat_df)
    order = np.argsort(-rr_scores)
    picked = np.asarray(
        [int(top_idx[j]) for j in order[:top_k]], dtype=np.int64
    )
    scores = np.asarray(
        [float(rr_scores[j]) for j in order[:top_k]], dtype=np.float64
    )
    return picked, scores


def save_reranker_bundle(bundle: ReRankerBundle, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(bundle, f)


def load_reranker_bundle(path: Path) -> ReRankerBundle | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, ReRankerBundle):
        return None
    return obj
