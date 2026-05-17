"""
VinylIQ price microservice.

Run from repository root (example):

  PYTHONPATH=. uvicorn price_estimator.src.api.main:app \\
      --host 0.0.0.0 --port 8801
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException

from price_estimator.src.api.schemas import (
    CollectionValueRequest,
    CollectionValueResponse,
    EstimateRequest,
    EstimateResponse,
    HealthResponse,
    InvalidateMarketplaceCacheResponse,
)
from price_estimator.src.inference.service import load_service_from_config

_cfg = os.environ.get("VINYLIQ_CONFIG")
CONFIG_PATH = Path(_cfg) if _cfg else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model + stores before accepting traffic so /health stays fast under
    # single-worker uvicorn (probes were timing out during cold get_service()).
    get_service()
    yield


app = FastAPI(
    title="VinylIQ Price API",
    description="ML-assisted vinyl price estimates for Discogs releases",
    version="0.1.0",
    lifespan=lifespan,
)

_svc = None


def get_service():
    global _svc
    if _svc is None:
        _svc = load_service_from_config(
            CONFIG_PATH if CONFIG_PATH and CONFIG_PATH.exists() else None
        )
    return _svc


def _check_api_key(x_api_key: str | None) -> None:
    expected = os.environ.get("VINYLIQ_API_KEY")
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


@app.get("/health", response_model=HealthResponse)
async def health(x_api_key: str | None = Header(None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    svc = get_service()
    # COUNT(*) can exceed kube probes on multi-million-row Postgres; ping only.
    svc.features.ping()
    md = svc.model_dir
    loaded = (
        (md / "model_manifest.json").is_file()
        or (md / "regressor.joblib").is_file()
        or (md / "xgb_model.joblib").is_file()
    )
    return HealthResponse(
        status="ok",
        feature_store_count=None,
        model_loaded=loaded,
        model_source=svc.model_source,
    )


@app.post("/estimate", response_model=EstimateResponse)
async def estimate(
    body: EstimateRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)
    svc = get_service()
    overlay = (
        body.marketplace_client.model_dump(exclude_none=True)
        if body.marketplace_client
        else None
    )
    if not overlay:
        overlay = None
    out = svc.estimate(
        body.release_id,
        body.media_condition,
        body.sleeve_condition,
        refresh_stats=body.refresh_stats,
        marketplace_client=overlay,
    )
    return EstimateResponse(**out)


@app.delete(
    "/cache/marketplace/{release_id}",
    response_model=InvalidateMarketplaceCacheResponse,
)
async def invalidate_marketplace_redis_cache(
    release_id: str,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """
    Drop the Redis L1 projection for this release_id.

    The next ``/estimate`` misses Redis, loads marketplace data from the backing
    store, and may repopulate Redis. Send ``refresh_stats: true`` on a later
    request to skip the store and refetch Discogs (releases + price_suggestions).
    """
    _check_api_key(x_api_key)
    svc = get_service()
    out = svc.invalidate_marketplace_redis_cache(release_id)
    return InvalidateMarketplaceCacheResponse(**out)


@app.post("/collection/value", response_model=CollectionValueResponse)
async def collection_value(
    body: CollectionValueRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """
    Sum estimates for a list of items.

    Does not fetch the user's Discogs collection; the client passes items.
    """
    _check_api_key(x_api_key)
    svc = get_service()
    items = [it.model_dump() for it in body.items]
    out = svc.estimate_batch(items)
    return CollectionValueResponse(**out)
