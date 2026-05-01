"""
VinylIQ price microservice.

Run from repository root:
  PYTHONPATH=. uvicorn price_estimator.src.api.main:app --host 0.0.0.0 --port 8801
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
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health", response_model=HealthResponse)
async def health(x_api_key: str | None = Header(None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    svc = get_service()
    # COUNT(*) can exceed kube probes on multi-million-row Postgres; ping only.
    svc.features.ping()
    loaded = (svc.model_dir / "xgb_model.joblib").exists()
    return HealthResponse(
        status="ok",
        feature_store_count=None,
        model_loaded=loaded,
    )


@app.post("/estimate", response_model=EstimateResponse)
async def estimate(
    body: EstimateRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)
    svc = get_service()
    out = svc.estimate(
        body.release_id,
        body.media_condition,
        body.sleeve_condition,
        refresh_stats=body.refresh_stats,
    )
    return EstimateResponse(**out)


@app.post("/collection/value", response_model=CollectionValueResponse)
async def collection_value(
    body: CollectionValueRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """
    Sum estimates for a list of items. Does not fetch the user's Discogs collection;
    the client must pass items (extension / app supplies them).
    """
    _check_api_key(x_api_key)
    svc = get_service()
    items = [it.model_dump() for it in body.items]
    out = svc.estimate_batch(items)
    return CollectionValueResponse(**out)
