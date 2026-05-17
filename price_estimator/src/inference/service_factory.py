"""Build :class:`InferenceService` from merged VinylIQ YAML.

Used by :func:`price_estimator.src.inference.service.load_service_from_config`
and callers that already hold a merged ``cfg`` dict (tests, advanced wiring).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .service import InferenceService


def build_inference_service_from_merged_config(
    cfg: dict[str, Any],
) -> "InferenceService":
    """Construct ``InferenceService`` from a fully merged VinylIQ config dict."""
    from .service import InferenceService, yaml_inference_condition_overlay

    root = Path(__file__).resolve().parents[2]
    v = cfg.get("vinyliq") or {}
    paths = v.get("paths") or {}
    ml_cfg = cfg.get("mlflow") or {}

    env_ml_uri = (os.environ.get("VINYLIQ_MLFLOW_MODEL_URI") or "").strip()
    _yaml_ml = v.get("mlflow_model_uri")
    yaml_ml_uri = str(_yaml_ml).strip() if _yaml_ml else ""
    ml_uri = env_ml_uri or yaml_ml_uri

    model_source = "local"
    md: Path
    if ml_uri:
        from ..mlflow_tracking import (
            apply_google_credentials_from_mlflow_config,
            resolve_mlflow_tracking_uri,
        )
        from .mlflow_bundle import download_vinyliq_bundle_to_cache

        apply_google_credentials_from_mlflow_config(ml_cfg)
        tracking_uri = resolve_mlflow_tracking_uri(ml_cfg)
        cache_env = (os.environ.get("VINYLIQ_MLFLOW_CACHE_DIR") or "").strip()
        _yc = paths.get("mlflow_cache_dir")
        cache_yaml = str(_yc).strip() if _yc else ""
        cache_raw = cache_env or cache_yaml
        cache_root = (
            Path(cache_raw)
            if cache_raw
            else root / "artifacts" / "mlflow_model_cache"
        )
        if not cache_root.is_absolute():
            cache_root = root / cache_root
        _fr = (os.environ.get("VINYLIQ_MLFLOW_FORCE_REFRESH") or "").strip().lower()
        force = _fr in ("1", "true", "yes")

        md = download_vinyliq_bundle_to_cache(
            model_uri=ml_uri,
            tracking_uri=tracking_uri,
            cache_root=cache_root,
            force=force,
        )
        model_source = "mlflow"
    else:
        md = Path(paths.get("model_dir", root / "artifacts" / "vinyliq"))
        if not md.is_absolute():
            md = root / md
    key = v.get("discogs_token_env", "DISCOGS_USER_TOKEN")
    explicit = (os.environ.get(key) or "").strip() if key else ""

    _tl = v.get("training_label")
    tl_raw = _tl if isinstance(_tl, dict) else {}
    yaml_overlay = yaml_inference_condition_overlay(cfg)
    nm_raw = tl_raw.get("price_suggestion_grade")
    nm_grade_key = (
        str(nm_raw).strip()
        if nm_raw is not None and str(nm_raw).strip()
        else "Near Mint (NM or M-)"
    )

    inf = v.get("inference") if isinstance(v.get("inference"), dict) else {}
    env_ps = (
        os.environ.get("VINYLIQ_USE_PRICE_SUGGESTION_CONDITION_ANCHOR") or ""
    ).strip().lower()
    if env_ps in ("1", "true", "yes"):
        use_ps_anchor = True
    elif env_ps in ("0", "false", "no"):
        use_ps_anchor = False
    else:
        use_ps_anchor = bool(
            inf.get("use_price_suggestion_condition_anchor", True)
        )

    from .anchor_guardrails_config import (
        AnchorGuardrailsConfig,
        anchor_guardrails_config_from_raw,
    )
    from ..training.sale_floor_blend_config import sale_floor_blend_config_from_raw

    ag_raw = inf.get("anchor_guardrails")
    ag_cfg = anchor_guardrails_config_from_raw(
        ag_raw if isinstance(ag_raw, dict) else None
    )
    env_ag = (os.environ.get("VINYLIQ_ANCHOR_GUARDRAILS") or "").strip().lower()
    if env_ag in ("1", "true", "yes"):
        ag_cfg = AnchorGuardrailsConfig(
            enabled=True,
            min_listing_usd=ag_cfg.min_listing_usd,
            min_listing_to_nm_rung_ratio=ag_cfg.min_listing_to_nm_rung_ratio,
            min_listing_to_max_rung_ratio=ag_cfg.min_listing_to_max_rung_ratio,
            inflated_max_rung_to_reference=ag_cfg.inflated_max_rung_to_reference,
            inflated_max_rung_to_sale_reference=ag_cfg.inflated_max_rung_to_sale_reference,
            max_ladder_to_reference=ag_cfg.max_ladder_to_reference,
            mint_outlier_multiple_of_nm=ag_cfg.mint_outlier_multiple_of_nm,
            ladder_rung_winsorize_multiple_of_nm=ag_cfg.ladder_rung_winsorize_multiple_of_nm,
        )
    elif env_ag in ("0", "false", "no"):
        ag_cfg = AnchorGuardrailsConfig(
            enabled=False,
            min_listing_usd=ag_cfg.min_listing_usd,
            min_listing_to_nm_rung_ratio=ag_cfg.min_listing_to_nm_rung_ratio,
            min_listing_to_max_rung_ratio=ag_cfg.min_listing_to_max_rung_ratio,
            inflated_max_rung_to_reference=ag_cfg.inflated_max_rung_to_reference,
            inflated_max_rung_to_sale_reference=ag_cfg.inflated_max_rung_to_sale_reference,
            max_ladder_to_reference=ag_cfg.max_ladder_to_reference,
            mint_outlier_multiple_of_nm=ag_cfg.mint_outlier_multiple_of_nm,
            ladder_rung_winsorize_multiple_of_nm=ag_cfg.ladder_rung_winsorize_multiple_of_nm,
        )

    sf_raw = tl_raw.get("sale_floor_blend")
    blend_cfg = sale_floor_blend_config_from_raw(
        sf_raw if isinstance(sf_raw, dict) else {},
        nm_grade_key=nm_grade_key,
    )

    dsn_key = paths.get("postgres_dsn_env")
    if dsn_key:
        dsn = (os.environ.get(str(dsn_key).strip()) or "").strip()
        if dsn:
            from ..storage.postgres_feature_store import PostgresFeatureStore
            from ..storage.postgres_marketplace_stats import PostgresMarketplaceStats

            return InferenceService(
                model_dir=md,
                marketplace_store=PostgresMarketplaceStats(dsn),
                feature_store=PostgresFeatureStore(dsn),
                discogs_token=explicit or None,
                model_source=model_source,
                nm_grade_key=nm_grade_key,
                yaml_condition_overlay=yaml_overlay,
                use_price_suggestion_condition_anchor=use_ps_anchor,
                anchor_guardrails_cfg=ag_cfg,
                sale_floor_blend_cfg=blend_cfg,
            )

    _mdb = root / "data" / "cache" / "marketplace_stats.sqlite"
    _fsdb = root / "data" / "feature_store.sqlite"
    mp = Path(paths.get("marketplace_db", _mdb))
    fs = Path(paths.get("feature_store_db", _fsdb))
    if not mp.is_absolute():
        mp = root / mp
    if not fs.is_absolute():
        fs = root / fs
    return InferenceService(
        marketplace_db=mp,
        feature_store_db=fs,
        model_dir=md,
        discogs_token=explicit or None,
        model_source=model_source,
        nm_grade_key=nm_grade_key,
        yaml_condition_overlay=yaml_overlay,
        use_price_suggestion_condition_anchor=use_ps_anchor,
        anchor_guardrails_cfg=ag_cfg,
        sale_floor_blend_cfg=blend_cfg,
    )
