"""VinylIQ inference: stats + feature store + XGBoost + condition adjustment."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from ..features.vinyliq_features import (
    MAX_LOG_PRICE,
    clamp_ordinals_for_inference,
    condition_string_to_ordinal,
    default_feature_columns,
    first_artist_id,
    first_label_id,
    grade_delta_scale_params_from_cond,
    row_dict_for_inference,
    scaled_condition_log_adjustment,
)
from ..models.condition_adjustment import (
    default_params,
    load_params_with_grade_delta_overlays,
    merge_inference_condition_params,
)
from ..models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    FittedVinylIQRegressor,
    load_fitted_regressor,
)
from ..storage.feature_store import FeatureStoreDB
from ..storage.marketplace_db import (
    MarketplaceStatsDB,
    decode_redis_marketplace_cached_payload,
    marketplace_inference_stats_from_row,
    merge_marketplace_client_overlay,
    redis_marketplace_cache_blob_from_row,
)
from ..storage.redis_stats_cache import RedisStatsCache
from ..training.sale_floor_targets import (
    inference_price_suggestion_anchor_usd_for_side,
    inference_residual_anchor_usd,
)

_MIN_PRICE_USD = 0.50
_MIN_RELEASE_YEAR = 1877
_MAX_RELEASE_YEAR = 2030


def _repo_root() -> Path:
    # price_estimator/src/inference/service.py -> parents[2] = price_estimator
    return Path(__file__).resolve().parents[2]


def _workspace_root_for_yaml_inherits() -> Path:
    """Monorepo root (parent of ``price_estimator``); Docker image layout is ``/app``."""
    return _repo_root().parent


def _deep_merge_vinyliq_yaml(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            _deep_merge_vinyliq_yaml(base[k], v)
        else:
            base[k] = v


def _load_yaml_with_inherits_file(
    path: Path,
    *,
    workspace_root: Path | None,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {}
    parent_raw = cfg.pop("inherits", None)
    ws = workspace_root if workspace_root is not None else _workspace_root_for_yaml_inherits()
    if parent_raw:
        raw = str(parent_raw).strip()
        if raw:
            parent_path = Path(raw)
            if not parent_path.is_absolute():
                parent_path = ws / parent_path
            base = (
                _load_yaml_with_inherits_file(parent_path, workspace_root=ws)
                if parent_path.is_file()
                else {}
            )
            _deep_merge_vinyliq_yaml(base, cfg)
            return base
    return cfg


def yaml_inference_condition_overlay(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """
    Read ``ordinal_cascade.{condition_adjustment,grade_delta_scale}`` into a dict
    suitable for :func:`merge_inference_condition_params`.

    Mirrors the training nesting under ``vinyliq.training_label.sale_floor_blend``.
    """
    v = cfg.get("vinyliq")
    if not isinstance(v, dict):
        return None
    tl = v.get("training_label")
    if not isinstance(tl, dict):
        return None
    sf = tl.get("sale_floor_blend")
    if not isinstance(sf, dict):
        return None
    oc = sf.get("ordinal_cascade")
    if not isinstance(oc, dict):
        return None

    out: dict[str, Any] = {}
    ca = oc.get("condition_adjustment")
    if isinstance(ca, dict):
        for k in ("alpha", "beta", "ref_grade"):
            if k not in ca or ca[k] is None:
                continue
            try:
                out[k] = float(ca[k])
            except (TypeError, ValueError):
                continue

    gds = oc.get("grade_delta_scale")
    if isinstance(gds, dict) and gds:
        out["grade_delta_scale"] = dict(gds)

    return out or None


def _ensure_shared_path() -> None:
    root = _repo_root().parent
    shared = root / "shared"
    if shared.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_yaml_config(
    path: Path | None = None,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """
    Load VinylIQ YAML. When ``inherits:`` is set (e.g. GKE ``price-config``), merge the
    parent chain (**parent → child**) so bundled ``configs/base.yaml`` shipped with the
    Docker image (under the monorepo root, ``/app`` in the cluster image) participates.
    """
    root_pkg = _repo_root()
    env = os.environ.get("VINYLIQ_CONFIG")
    p_raw = path or (Path(env) if env else root_pkg / "configs" / "base.yaml")
    p = Path(p_raw)
    merged = _load_yaml_with_inherits_file(
        p,
        workspace_root=workspace_root,
    )
    return merged if isinstance(merged, dict) else {}


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
        model_dir: Path,
        marketplace_db: Path | None = None,
        feature_store_db: Path | None = None,
        marketplace_store: Any | None = None,
        feature_store: Any | None = None,
        discogs_token: str | None = None,
        genre_encoder_path: Path | None = None,
        redis_cache: RedisStatsCache | None = None,
        model_source: str = "local",
        nm_grade_key: str = "Near Mint (NM or M-)",
        yaml_condition_overlay: dict[str, Any] | None = None,
        use_price_suggestion_condition_anchor: bool = True,
    ) -> None:
        if marketplace_store is not None:
            self.marketplace = marketplace_store
        elif marketplace_db is not None:
            self.marketplace = MarketplaceStatsDB(marketplace_db)
        else:
            raise ValueError("InferenceService requires marketplace_store or marketplace_db")

        if feature_store is not None:
            self.features = feature_store
        elif feature_store_db is not None:
            self.features = FeatureStoreDB(feature_store_db)
        else:
            raise ValueError("InferenceService requires feature_store or feature_store_db")

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
        # Optional L1 read cache. Construction never raises; caller may pass
        # ``RedisStatsCache()`` and the host comes from REDIS_HOST. Disabled
        # cache is a no-op fallthrough to SQLite (the existing behavior).
        self.redis_cache = redis_cache if redis_cache is not None else RedisStatsCache()
        self.model_source = str(model_source).strip().lower() or "local"
        self._nm_grade_key = str(nm_grade_key).strip() or "Near Mint (NM or M-)"
        self._yaml_condition_overlay = yaml_condition_overlay or None
        self._use_ps_condition_anchor = bool(use_price_suggestion_condition_anchor)

    def _effective_condition_params(self) -> dict[str, Any]:
        base = load_params_with_grade_delta_overlays(self.model_dir)
        return merge_inference_condition_params(base, self._yaml_condition_overlay)

    def _residual_price_single_ps_path(
        self,
        *,
        ladder_side: Literal["media", "sleeve"],
        logp_raw: float,
        stats: dict[str, Any],
        baseline,
        media_condition: str | None,
        sleeve_condition: str | None,
        media_ord: float,
        sleeve_ord: float,
        cond_params: dict[str, Any],
        release_year: float | None,
        scale_p,
    ) -> tuple[float, float]:
        """
        One residual reconstruction: ladder anchor from either media or sleeve grade,
        fallback to ``inference_residual_anchor_usd`` / listing; suppress ordinal drift
        when the PS ladder rung anchored the path.

        Returns (price_usd, anchor_usd_used).
        """
        anchor_f: float | None = None
        use_ps_path = False
        ps_anchor = inference_price_suggestion_anchor_usd_for_side(
            stats,
            role=ladder_side,
            media_condition=media_condition,
            sleeve_condition=sleeve_condition,
        )
        if ps_anchor is not None and ps_anchor > 0.0:
            anchor_f = float(ps_anchor)
            use_ps_path = True
        if anchor_f is None:
            anchor_f = inference_residual_anchor_usd(
                stats,
                nm_grade_key=self._nm_grade_key,
            )
            if anchor_f is None or anchor_f <= 0.0:
                mp_b = baseline
                anchor_f = (
                    float(mp_b)
                    if mp_b is not None and float(mp_b) > 0
                    else 0.0
                )
        anchor = float(anchor_f)
        logp = logp_raw + float(np.log1p(max(anchor, 0.0)))
        _dp = default_params()
        anchor_scale = float(anchor) if anchor > 0 else 1.0
        ref_grade_adj = clamp_ordinals_for_inference(
            float(cond_params.get("ref_grade", _dp["ref_grade"])),
            float(cond_params.get("ref_grade", _dp["ref_grade"])),
        )[0]
        media_ord_adj, sleeve_ord_adj = media_ord, sleeve_ord
        if use_ps_path:
            media_ord_adj, sleeve_ord_adj = clamp_ordinals_for_inference(
                ref_grade_adj, ref_grade_adj
            )
        logp_adj = scaled_condition_log_adjustment(
            logp,
            media_ord_adj,
            sleeve_ord_adj,
            base_alpha=float(cond_params.get("alpha", _dp["alpha"])),
            base_beta=float(cond_params.get("beta", _dp["beta"])),
            ref_grade=float(cond_params.get("ref_grade", 8.0)),
            anchor_usd=max(anchor_scale, 1e-6),
            release_year=release_year,
            scale_params=scale_p,
        )
        raw_price = float(np.expm1(np.clip(logp_adj, 0, MAX_LOG_PRICE)))
        price = max(raw_price, _MIN_PRICE_USD)
        return price, anchor

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

    def invalidate_marketplace_redis_cache(self, release_id: str) -> dict[str, Any]:
        """Remove ``vinyliq:marketplace:stats:{release_id}`` so the next fetch repopulates.

        Postgres/SQLite rows are untouched. When Redis is disabled, this is a no-op.

        Returns a small summary for APIs (does not indicate whether DEL matched a key).
        """
        rid = str(release_id).strip()
        enabled_before = self.redis_cache.enabled()
        self.redis_cache.invalidate(rid)
        return {
            "release_id": rid,
            "redis_cache_enabled": enabled_before,
        }

    def fetch_stats(
        self,
        release_id: str,
        *,
        use_cache: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Read-through cache: Redis -> persistent store (Postgres/SQLite) -> Discogs API.

        Live hydrate matches ``scripts/collect_marketplace_stats.py`` **full** mode
        (same two calls per ``release_id``): ``GET /releases/{id}`` for listing/community
        fields (``extract_release_listing_fields``) and ``GET /marketplace/price_suggestions/{id}``
        for the grade ladder.

        Redis persists the same inference projection as the backing row (listing floor,
        depth, community counts, ladder when present).

        Older Redis entries with only ``release_lowest_price`` / ``num_for_sale`` are
        upgraded on read.
        """
        rid = str(release_id).strip()
        if use_cache and not refresh:
            cached = self.redis_cache.get(rid)
            if cached is not None:
                core = decode_redis_marketplace_cached_payload(cached)
                return {**core, "source": "cache_redis"}
            row = self.marketplace.get(rid)
            if row:
                blob = redis_marketplace_cache_blob_from_row(row)
                self.redis_cache.set(rid, blob)
                return {
                    **marketplace_inference_stats_from_row(row),
                    "source": "cache_db",
                }
        client = self._get_discogs_client()
        if not client:
            row = self.marketplace.get(rid)
            if row:
                blob = redis_marketplace_cache_blob_from_row(row)
                self.redis_cache.set(rid, blob)
                return {
                    **marketplace_inference_stats_from_row(row),
                    "source": "cache",
                }
            return {
                **marketplace_inference_stats_from_row({}),
                "source": "none",
            }
        release_pl: dict[str, Any] | None = None
        try:
            raw_rel = client.get_release_with_retries(
                rid,
                max_retries=4,
                backoff_base=1.5,
                backoff_max=60.0,
                timeout=45.0,
            )
        except Exception:
            release_pl = None
        else:
            release_pl = raw_rel if isinstance(raw_rel, dict) else None

        sugg_pl: dict[str, Any] = {}
        try:
            raw_sugg = client.get_price_suggestions_with_retries(
                rid,
                max_retries=4,
                backoff_base=1.5,
                backoff_max=60.0,
                timeout=45.0,
            )
            if isinstance(raw_sugg, dict):
                sugg_pl = raw_sugg
        except Exception:
            sugg_pl = {}

        self.marketplace.upsert(
            rid,
            {},
            release_payload=release_pl,
            price_suggestions_payload=sugg_pl,
        )
        row = self.marketplace.get(rid) or {}
        blob = redis_marketplace_cache_blob_from_row(row)
        self.redis_cache.set(rid, blob)
        return {**marketplace_inference_stats_from_row(row), "source": "live"}

    def estimate(
        self,
        release_id: str,
        media_condition: str | None,
        sleeve_condition: str | None,
        *,
        refresh_stats: bool = False,
        marketplace_client: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fetched = self.fetch_stats(release_id, refresh=refresh_stats)
        stats_flat = {k: v for k, v in fetched.items() if k != "source"}
        stats = merge_marketplace_client_overlay(stats_flat, marketplace_client)
        baseline = stats.get("release_lowest_price")
        nfs = int(stats.get("num_for_sale") or 0)
        warnings: list[str] = []
        if nfs < 3:
            warnings.append("low_market_depth")
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
            "Near Mint (NM or M-)",
            "Near Mint (NM or M-)",
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
        cond_params = self._effective_condition_params()
        media_ord = condition_string_to_ordinal(media_condition)
        sleeve_ord = condition_string_to_ordinal(sleeve_condition)
        media_ord, sleeve_ord = clamp_ordinals_for_inference(media_ord, sleeve_ord)

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
                "num_for_sale": nfs,
                "warnings": warnings,
            }

        logp_raw = float(model.predict_log1p(x)[0])

        yr_raw = (cat or {}).get("year")
        try:
            release_year = float(yr_raw) if yr_raw is not None else None
        except (TypeError, ValueError):
            release_year = None
        if release_year is not None and not np.isfinite(release_year):
            release_year = None
        if release_year is not None:
            release_year = float(
                max(_MIN_RELEASE_YEAR, min(_MAX_RELEASE_YEAR, release_year))
            )
        scale_p = grade_delta_scale_params_from_cond(cond_params)

        price: float
        residual_anchor: float | None

        if (
            model.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN
            and self._use_ps_condition_anchor
        ):
            pm, am = self._residual_price_single_ps_path(
                ladder_side="media",
                logp_raw=logp_raw,
                stats=stats,
                baseline=baseline,
                media_condition=media_condition,
                sleeve_condition=sleeve_condition,
                media_ord=media_ord,
                sleeve_ord=sleeve_ord,
                cond_params=cond_params,
                release_year=release_year,
                scale_p=scale_p,
            )
            ps, aa = self._residual_price_single_ps_path(
                ladder_side="sleeve",
                logp_raw=logp_raw,
                stats=stats,
                baseline=baseline,
                media_condition=media_condition,
                sleeve_condition=sleeve_condition,
                media_ord=media_ord,
                sleeve_ord=sleeve_ord,
                cond_params=cond_params,
                release_year=release_year,
                scale_p=scale_p,
            )
            price = max((float(pm) + float(ps)) / 2.0, _MIN_PRICE_USD)
            residual_anchor = float((float(am) + float(aa)) / 2.0)
        else:
            anchor = 0.0
            residual_anchor = None
            if model.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
                anchor_f = inference_residual_anchor_usd(
                    stats,
                    nm_grade_key=self._nm_grade_key,
                )
                if anchor_f is None or anchor_f <= 0.0:
                    mp_b = baseline
                    anchor_f = (
                        float(mp_b)
                        if mp_b is not None and float(mp_b) > 0
                        else 0.0
                    )
                anchor = float(anchor_f)
                residual_anchor = anchor
                logp = logp_raw + float(np.log1p(max(anchor, 0.0)))
            else:
                logp = logp_raw
                mp2 = stats.get("release_lowest_price")
                if mp2 is not None and float(mp2) > 0:
                    anchor = float(mp2)

            anchor_scale = float(anchor) if anchor > 0 else 1.0
            _dp = default_params()
            logp_adj = scaled_condition_log_adjustment(
                logp,
                media_ord,
                sleeve_ord,
                base_alpha=float(cond_params.get("alpha", _dp["alpha"])),
                base_beta=float(cond_params.get("beta", _dp["beta"])),
                ref_grade=float(cond_params.get("ref_grade", 8.0)),
                anchor_usd=max(anchor_scale, 1e-6),
                release_year=release_year,
                scale_params=scale_p,
            )
            price = float(np.expm1(np.clip(logp_adj, 0, MAX_LOG_PRICE)))
            price = max(price, _MIN_PRICE_USD)
        spread = max(price * 0.12, 1.0)
        result: dict[str, Any] = {
            "release_id": str(release_id),
            "estimated_price": round(price, 2),
            "confidence_interval": [
                round(max(0.0, price - spread), 2),
                round(price + spread, 2),
            ],
            "baseline_median": round(float(baseline), 2) if baseline else None,
            "model_version": f"vinyliq_{model.backend}_v1",
            "status": "ok",
            "num_for_sale": nfs,
            "warnings": warnings,
        }
        if (
            model.target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN
            and residual_anchor is not None
            and residual_anchor > 0.0
        ):
            result["residual_anchor_usd"] = round(float(residual_anchor), 2)
        return result

    def estimate_batch(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        breakdown = []
        total = 0.0
        for it in items:
            rid = str(it.get("release_id", ""))
            mc = it.get("marketplace_client")
            overlay = mc if isinstance(mc, dict) else None
            out = self.estimate(
                rid,
                it.get("media_condition"),
                it.get("sleeve_condition"),
                marketplace_client=overlay,
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
    from .service_factory import build_inference_service_from_merged_config

    return build_inference_service_from_merged_config(cfg)
