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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..retrieval.candidates import RetrievalMetadata


# Features whose presence in a pickled bundle signals a pre-cleanup schema.
_LEGACY_RERANKER_FEATURES: frozenset[str] = frozenset(
    {"als_score", "item_popularity", "item_distinct_users", "item_priority"}
)


@dataclass
class ReRankerConfig:
    enabled: bool = False
    model_type: str = "linear"  # linear | pointwise
    candidate_top_n: int = 500
    train_sample_n: int = 250000
    negative_sampling: int = 200
    class_weight: str | None = "balanced"
    hard_negative_ratio: float = 0.7
    # Fraction at the very top of ALS non-positives to skip when mining hard
    # negatives; those are the likeliest false negatives under leave-one-out.
    hard_negative_skip_top_frac: float = 0.1
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
        hard_negative_skip_top_frac=float(
            d.get("hard_negative_skip_top_frac", 0.1)
        ),
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


_USER_TOP_GENRES_K = 10


def _discogs_rerank_features(aid: str, meta: RetrievalMetadata) -> dict[str, float]:
    """Tier A + Tier B Discogs-side scalars for one candidate album_id."""
    rc = int(getattr(meta, "album_release_count", {}).get(aid, 0) or 0)
    vr = int(getattr(meta, "album_vinyl_release_count", {}).get(aid, 0) or 0)
    uc = int(getattr(meta, "album_unique_country_count", {}).get(aid, 0) or 0)
    ul = int(getattr(meta, "album_unique_label_count", {}).get(aid, 0) or 0)
    span = int(getattr(meta, "album_era_span", {}).get(aid, 0) or 0)
    span = max(0, min(80, span))
    has_dm = (
        1.0 if getattr(meta, "album_has_discogs_master", {}).get(aid, False) else 0.0
    )

    cw = int(getattr(meta, "album_community_want", {}).get(aid, 0) or 0)
    ch = int(getattr(meta, "album_community_have", {}).get(aid, 0) or 0)
    nfs = int(getattr(meta, "album_num_for_sale", {}).get(aid, 0) or 0)
    lp = float(getattr(meta, "album_lowest_price", {}).get(aid, 0.0) or 0.0)
    has_cs = (
        1.0
        if getattr(meta, "album_has_community_stats", {}).get(aid, False)
        else 0.0
    )

    ratio = float(cw) / float(max(1, ch))
    ratio = max(0.0, min(10.0, ratio))

    return {
        "release_count_log": float(np.log1p(rc)),
        "vinyl_release_count_log": float(np.log1p(vr)),
        "unique_country_count": float(uc),
        "unique_label_count_log": float(np.log1p(ul)),
        "era_span": float(span),
        "has_discogs_master": has_dm,
        "discogs_community_want_log": float(np.log1p(cw)),
        "discogs_community_have_log": float(np.log1p(ch)),
        "discogs_want_have_ratio": ratio,
        "discogs_num_for_sale_log": float(np.log1p(nfs)),
        "discogs_lowest_price_log": float(np.log1p(max(0.0, lp))),
        "has_community_stats": has_cs,
    }


def _feature_rows(
    *,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    item_ids: np.ndarray,
    train_albums: set[str],
    meta: RetrievalMetadata,
) -> list[dict[str, float]]:
    # User genre + artist profile. We compute counts once per user and reuse
    # them across the full candidate pool: the cost is O(|train_albums|),
    # dominated by the per-candidate feature loop.
    user_genre_counts: dict[str, int] = {}
    user_artist_counts: dict[str, int] = {}
    years: list[int] = []
    user_ratings: list[float] = []
    for a in train_albums:
        aid = str(a)
        for g in meta.album_genres.get(aid, set()):
            user_genre_counts[g] = user_genre_counts.get(g, 0) + 1
        art = meta.album_id_to_artist.get(aid, "")
        if art:
            user_artist_counts[art] = user_artist_counts.get(art, 0) + 1
        y = meta.album_year.get(aid, 0)
        if y > 0:
            years.append(int(y))
        r = meta.album_avg_rating.get(aid, 0.0)
        if r > 0:
            user_ratings.append(float(r))

    user_genres: set[str] = set(user_genre_counts.keys())
    user_top_genres: set[str] = set(
        sorted(user_genre_counts, key=user_genre_counts.get, reverse=True)[
            :_USER_TOP_GENRES_K
        ]
    )
    total_genre_mass = sum(user_genre_counts.values()) or 1
    user_artist_set = set(user_artist_counts.keys())
    user_year = float(np.mean(years)) if years else 0.0
    # np.std over <2 samples returns 0; treat that as "single-era user" and
    # let the zdist feature fall back to the raw year distance.
    user_year_std = float(np.std(years)) if len(years) >= 2 else 0.0
    user_rating_mean = float(np.mean(user_ratings)) if user_ratings else 0.0
    user_n_train = len(train_albums)
    user_activity_log = float(np.log1p(user_n_train))

    # Per-user z-score of ALS scores over the candidate pool. Removes the
    # ||user_factor|| scale leak that raw als_score had across users.
    if top_scores.size > 0:
        mu = float(np.mean(top_scores))
        sd = float(np.std(top_scores)) or 1.0
    else:
        mu, sd = 0.0, 1.0

    rows: list[dict[str, float]] = []
    for rank0, (idx, als_s) in enumerate(zip(top_idx, top_scores)):
        aid = str(item_ids[int(idx)])
        cg = meta.album_genres.get(aid, set())
        c_artist = meta.album_id_to_artist.get(aid, "")
        c_year = float(meta.album_year.get(aid, 0))
        c_rating = float(meta.album_avg_rating.get(aid, 0.0))
        genre_mass_matched = sum(user_genre_counts.get(g, 0) for g in cg)
        year_valid = user_year > 0 and c_year > 0
        rows.append(
            {
                "als_score_z": (float(als_s) - mu) / sd,
                "als_rank_inv": float(1.0 / (1.0 + rank0)),
                "genre_jaccard": _genre_jaccard(user_genres, cg),
                "artist_match": (
                    1.0 if c_artist and c_artist in user_artist_set else 0.0
                ),
                "year_distance": (
                    abs(c_year - user_year) if year_valid else 0.0
                ),
                "item_avg_rating": c_rating,
                "user_top_genre_jaccard": _genre_jaccard(user_top_genres, cg),
                "user_genre_affinity": (
                    float(genre_mass_matched) / float(total_genre_mass)
                ),
                "user_artist_affinity": float(
                    np.log1p(user_artist_counts.get(c_artist, 0))
                    if c_artist
                    else 0.0
                ),
                "user_year_zdist": (
                    abs(c_year - user_year) / max(1.0, user_year_std)
                    if year_valid
                    else 0.0
                ),
                "item_rating_vs_user_mean": (
                    c_rating - user_rating_mean
                    if user_rating_mean > 0 and c_rating > 0
                    else 0.0
                ),
                "user_activity_log": user_activity_log,
                **_discogs_rerank_features(aid, meta),
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
        )
        u_rows: list[dict[str, float | int | str]] = []
        for i, fr in enumerate(feat_rows):
            idx = int(top_idx[i])
            label = 1 if idx in relevant_idx else 0
            u_rows.append(
                {"user_id": uid, "item_idx": idx, "label": label, **fr}
            )
        pos_rows = [r for r in u_rows if int(r["label"]) == 1]
        # neg_rows preserves ALS rank order (desc) from topn_als_from_candidates.
        neg_rows = [r for r in u_rows if int(r["label"]) == 0]
        if not pos_rows:
            users_no_pos += 1
        keep_neg = min(len(neg_rows), rr_cfg.negative_sampling)
        if keep_neg > 0:
            n_neg = len(neg_rows)
            # Skip the very top of ALS non-positives: under leave-one-out those
            # are the likeliest false negatives. Mine hard negatives from a
            # mid-rank band so the model does not learn "high ALS -> negative".
            skip = int(rr_cfg.hard_negative_skip_top_frac * n_neg)
            mid_end = int(0.5 * n_neg)
            hard_band = neg_rows[skip:mid_end] if mid_end > skip else []
            hard_n = int(round(keep_neg * rr_cfg.hard_negative_ratio))
            if hard_band and hard_n > 0:
                ix = rng.choice(
                    len(hard_band),
                    size=min(hard_n, len(hard_band)),
                    replace=False,
                )
                hard = [
                    hard_band[int(j)] for j in np.asarray(ix).ravel().tolist()
                ]
            else:
                hard = []
            tail = neg_rows[mid_end:]  # random tail (easy negatives)
            random_keep: list[dict[str, float | int | str]] = []
            remaining = keep_neg - len(hard)
            if remaining > 0 and tail:
                kk = min(remaining, len(tail))
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
        "als_score_z",
        "als_rank_inv",
        "genre_jaccard",
        "artist_match",
        "year_distance",
        "item_avg_rating",
        "user_top_genre_jaccard",
        "user_genre_affinity",
        "user_artist_affinity",
        "user_year_zdist",
        "item_rating_vs_user_mean",
        "user_activity_log",
        "release_count_log",
        "vinyl_release_count_log",
        "unique_country_count",
        "unique_label_count_log",
        "era_span",
        "has_discogs_master",
        "discogs_community_want_log",
        "discogs_community_have_log",
        "discogs_want_have_ratio",
        "discogs_num_for_sale_log",
        "discogs_lowest_price_log",
        "has_community_stats",
    ]
    X = train_df[feature_names].astype(float).values
    y = train_df["label"].astype(int).values
    if np.unique(y).shape[0] < 2:
        return None

    if rr_cfg.model_type == "linear":
        # Wrap LogisticRegression in a StandardScaler Pipeline so features on
        # different natural scales (als_score_z, item_avg_rating, year_distance)
        # all contribute comparably to the linear decision boundary.
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=250,
                        class_weight=rr_cfg.class_weight,
                        random_state=rr_cfg.random_state,
                    ),
                ),
            ]
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
    need = list(bundle.feature_names)
    missing = [c for c in need if c not in feat_df.columns]
    if missing:
        feat_df = feat_df.copy()
        for c in missing:
            feat_df[c] = 0.0
    X = feat_df[need].astype(float).values
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
    # Backward-compat: a bundle pickled before the feature cleanup carries
    # names that _feature_rows no longer emits. Refuse to load so serving
    # cleanly falls back to ALS-only instead of KeyError-ing at predict time.
    legacy = _LEGACY_RERANKER_FEATURES.intersection(obj.feature_names)
    if legacy:
        print(
            "reranker.pkl predates the feature cleanup "
            f"(legacy features: {sorted(legacy)}); "
            "re-run recommender.pipeline to regenerate."
        )
        return None
    return obj
