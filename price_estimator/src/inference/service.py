"""VinylIQ inference: stats + feature store + XGBoost + condition adjustment."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..features.vinyliq_features import (
    apply_condition_log_adjustment,
    condition_string_to_ordinal,
    default_feature_columns,
    first_artist_id,
    first_label_id,
    row_dict_for_inference,
)
from ..models.condition_adjustment import load_params
from ..models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
    load_fitted_regressor,
)
from ..storage.feature_store import FeatureStoreDB
from ..storage.marketplace_db import MarketplaceStatsDB


def _repo_root() -> Path:
    # price_estimator/src/inference/service.py -> parents[2] = price_estimator
    return Path(__file__).resolve().parents[2]


def _ensure_shared_path() -> None:
    root = _repo_root().parent
    shared = root / "shared"
    if shared.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    import os

    root = _repo_root()
    env = os.environ.get("VINYLIQ_CONFIG")
    p = path or (Path(env) if env else root / "configs" / "base.yaml")
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _parse_encoder_json(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Detect nested catalog_encoders vs legacy flat genre map."""
    if not raw:
        return {}
    first_v = next(iter(raw.values()))
    if isinstance(first_v, dict):
        out: dict[str, dict[str, float]] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                out[str(k)] = {str(kk): float(vv) for kk, vv in v.items()}
        return out
    return {"genre": {str(k): float(v) for k, v in raw.items()}}


def load_catalog_encoders_for_model_dir(
    model_dir: Path,
    encoder_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    empty = {
        "genre": {},
        "country": {},
        "primary_artist_id": {},
        "primary_label_id": {},
    }
    candidates: list[Path] = []
    if encoder_path is not None:
        candidates.append(encoder_path)
    candidates.append(model_dir / "catalog_encoders.json")
    candidates.append(model_dir / "genre_encoder.json")
    for p in candidates:
        if not p.is_file():
            continue
        try:
            parsed = _parse_encoder_json(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
        if not parsed:
            continue
        return {
            "genre": parsed.get("genre", {}),
            "country": parsed.get("country", {}),
            "primary_artist_id": parsed.get("primary_artist_id", {}),
            "primary_label_id": parsed.get("primary_label_id", {}),
        }
    return empty


class InferenceService:
    def __init__(
        self,
        *,
        marketplace_db: Path,
        feature_store_db: Path,
        model_dir: Path,
        discogs_token: str | None = None,
        genre_encoder_path: Path | None = None,
    ) -> None:
        self.marketplace = MarketplaceStatsDB(marketplace_db)
        self.features = FeatureStoreDB(feature_store_db)
        self.model_dir = Path(model_dir)
        self._discogs_token_explicit = (
            str(discogs_token).strip() if discogs_token else None
        )
        self._model: FittedVinylIQRegressor | None = None
        ge = genre_encoder_path
        self._catalog_encoders = load_catalog_encoders_for_model_dir(
            self.model_dir,
            encoder_path=Path(ge) if ge else None,
        )

    def _get_discogs_client(self):
        _ensure_shared_path()
        from shared.discogs_api.client import DiscogsClient, discogs_client_from_env

        if self._discogs_token_explicit:
            return DiscogsClient(user_token=self._discogs_token_explicit)
        return discogs_client_from_env()

    def _load_model(self) -> FittedVinylIQRegressor | None:
        if self._model is not None:
            return self._model
        loaded = load_fitted_regressor(self.model_dir)
        if loaded is None:
            return None
        self._model = loaded
        return self._model

    def fetch_stats(
        self,
        release_id: str,
        *,
        use_cache: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        rid = str(release_id).strip()
        if use_cache and not refresh:
            row = self.marketplace.get(rid)
            if row:
                return {
                    "lowest_price": row["lowest_price"],
                    "median_price": row["median_price"],
                    "release_lowest_price": row.get("release_lowest_price"),
                    "num_for_sale": row["num_for_sale"],
                    "source": "cache",
                }
        client = self._get_discogs_client()
        if not client:
            row = self.marketplace.get(rid)
            if row:
                return {
                    "lowest_price": row["lowest_price"],
                    "median_price": row["median_price"],
                    "release_lowest_price": row.get("release_lowest_price"),
                    "num_for_sale": row["num_for_sale"],
                    "source": "cache",
                }
            return {
                "lowest_price": None,
                "median_price": None,
                "release_lowest_price": None,
                "num_for_sale": 0,
                "source": "none",
            }
        release_pl = None
        try:
            release_pl = client.get_release_with_retries(
                rid, max_retries=4, backoff_base=1.5, backoff_max=60.0, timeout=45.0
            )
        except Exception:
            release_pl = None
        if not isinstance(release_pl, dict):
            release_pl = None
        self.marketplace.upsert(
            rid,
            {},
            release_payload=release_pl,
        )
        row = self.marketplace.get(rid) or {}
        return {
            "lowest_price": row.get("lowest_price"),
            "median_price": row.get("median_price"),
            "release_lowest_price": row.get("release_lowest_price"),
            "num_for_sale": int(row.get("num_for_sale") or 0),
            "source": "live",
        }

    def estimate(
        self,
        release_id: str,
        media_condition: str | None,
        sleeve_condition: str | None,
        *,
        refresh_stats: bool = False,
    ) -> dict[str, Any]:
        stats = self.fetch_stats(release_id, refresh=refresh_stats)
        baseline = (
            stats.get("release_lowest_price")
            or stats.get("lowest_price")
            or stats.get("median_price")
        )
        cat = self.features.get(str(release_id).strip())
        enc = self._catalog_encoders
        genre = (cat or {}).get("genre") or ""
        gkey = str(genre).strip().lower()[:80]
        genre_index = float(enc.get("genre", {}).get(gkey, 0.0))
        ckey = str((cat or {}).get("country") or "").strip().lower()[:80]
        country_index = float(enc.get("country", {}).get(ckey, 0.0))
        pa = first_artist_id(cat or {})
        pl = first_label_id(cat or {})
        primary_artist_index = float(enc.get("primary_artist_id", {}).get(pa, 0.0))
        primary_label_index = float(enc.get("primary_label_id", {}).get(pl, 0.0))

        model = self._load_model()
        include_mkt = True
        if model is not None:
            include_mkt = model.target_kind != TARGET_KIND_RESIDUAL_LOG_MEDIAN

        row = row_dict_for_inference(
            str(release_id),
            media_condition,
            sleeve_condition,
            stats,
            cat,
            genre_index=genre_index,
            country_index=country_index,
            primary_artist_index=primary_artist_index,
            primary_label_index=primary_label_index,
            include_marketplace_scalars_in_features=include_mkt,
        )
        cols = list(model.feature_columns) if model is not None else default_feature_columns()
        x = np.array([[float(row[c]) for c in cols]], dtype=np.float64)
        cond_params = load_params(self.model_dir / "condition_params.json")
        media_ord = condition_string_to_ordinal(media_condition)
        sleeve_ord = condition_string_to_ordinal(sleeve_condition)

        if model is None:
            # Fallback: baseline only
            b = float(baseline) if baseline is not None else 0.0
            low = b * 0.85 if b else 0.0
            high = b * 1.15 if b else 0.0
            return {
                "release_id": str(release_id),
                "estimated_price": round(b, 2) if b else None,
                "confidence_interval": [round(low, 2), round(high, 2)] if b else [0.0, 0.0],
                "baseline_median": round(b, 2) if b else None,
                "model_version": "fallback_baseline",
                "status": "no_model",
            }

        logp_raw = float(model.predict_log1p(x)[0])
        if model.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            mp = (
                stats.get("release_lowest_price")
                or stats.get("lowest_price")
                or stats.get("median_price")
            )
            anchor = float(mp) if mp is not None and float(mp) > 0 else 0.0
            if anchor <= 0.0 and baseline is not None:
                anchor = float(baseline)
            logp = logp_raw + float(np.log1p(max(anchor, 0.0)))
        else:
            logp = logp_raw
        logp_adj = apply_condition_log_adjustment(
            logp,
            media_ord,
            sleeve_ord,
            alpha=float(cond_params.get("alpha", -0.06)),
            beta=float(cond_params.get("beta", -0.04)),
            ref_grade=float(cond_params.get("ref_grade", 8.0)),
        )
        price = float(np.expm1(np.clip(logp_adj, 0, 25)))
        spread = max(price * 0.12, 1.0)
        return {
            "release_id": str(release_id),
            "estimated_price": round(price, 2),
            "confidence_interval": [
                round(max(0.0, price - spread), 2),
                round(price + spread, 2),
            ],
            "baseline_median": round(float(baseline), 2) if baseline else None,
            "model_version": f"vinyliq_{model.backend}_v1",
            "status": "ok",
        }

    def estimate_batch(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        breakdown = []
        total = 0.0
        for it in items:
            rid = str(it.get("release_id", ""))
            out = self.estimate(
                rid,
                it.get("media_condition"),
                it.get("sleeve_condition"),
            )
            breakdown.append(out)
            if out.get("estimated_price") is not None:
                total += float(out["estimated_price"])
        return {
            "total_estimated_value": round(total, 2),
            "per_item_breakdown": breakdown,
        }


def load_service_from_config(config_path: Path | None = None) -> InferenceService:
    _ensure_shared_path()
    from shared.project_env import load_project_dotenv

    load_project_dotenv()
    cfg = load_yaml_config(config_path)
    root = _repo_root()
    v = cfg.get("vinyliq") or {}
    paths = v.get("paths") or {}
    mp = Path(paths.get("marketplace_db", root / "data" / "cache" / "marketplace_stats.sqlite"))
    fs = Path(paths.get("feature_store_db", root / "data" / "feature_store.sqlite"))
    md = Path(paths.get("model_dir", root / "artifacts" / "vinyliq"))
    if not mp.is_absolute():
        mp = root / mp
    if not fs.is_absolute():
        fs = root / fs
    if not md.is_absolute():
        md = root / md
    key = v.get("discogs_token_env", "DISCOGS_USER_TOKEN")
    explicit = (os.environ.get(key) or "").strip() if key else ""
    return InferenceService(
        marketplace_db=mp,
        feature_store_db=fs,
        model_dir=md,
        discogs_token=explicit or None,
    )
