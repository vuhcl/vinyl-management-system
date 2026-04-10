"""
ML component APIs: recommender, condition classifier, price estimator.

Endpoints (router prefix ``/api``):

- ``GET /recommendations`` — loads recommender artifacts from ``paths.artifacts``
  in ``core.config.load_config()`` (default ``artifacts/``). Requires
  ``als_model.pkl``. Run pipeline after ingest.
- ``POST /condition`` — baseline grader (TF-IDF + LR) via ``grader.src.pipeline``;
  503 when baseline artifacts are missing.
- ``GET /price/{release_id}`` — if ``PRICE_SERVICE_URL`` is set, proxies to
  ``POST {base}/estimate`` with optional ``VINYLIQ_API_KEY`` header; else
  ``price_estimator.src.pipeline.estimate()`` in-process.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from functools import lru_cache
from pydantic import BaseModel

from web.app.deps import get_current_username

router = APIRouter()

# Substrings in grader exceptions that usually mean missing baseline artifacts.
_BASELINE_ARTIFACT_ERROR_HINTS = frozenset(
    (
        "no such file",
        "not found",
        "model not calibrated",
        "pickle",
        "pkl",
        "vectorizer",
        "encoder",
        "features",
    )
)


def _web_price_response_from_service(
    data: dict,
    release_id: str,
    sleeve_condition: str | None,
    media_condition: str | None,
) -> dict:
    """
    Map VinylIQ ``POST /estimate`` JSON to the web API shape.

    ``estimated_price`` → ``estimate_usd``; ``confidence_interval`` [lo, hi] →
    ``interval_low`` / ``interval_high``; pass through ``baseline_median``,
    ``model_version``, ``status``.
    """
    ci = data.get("confidence_interval") or [None, None]
    return {
        "release_id": data.get("release_id", release_id),
        "sleeve_condition": sleeve_condition,
        "media_condition": media_condition,
        "estimate_usd": data.get("estimated_price"),
        "interval_low": ci[0] if len(ci) > 0 else None,
        "interval_high": ci[1] if len(ci) > 1 else None,
        "baseline_median": data.get("baseline_median"),
        "model_version": data.get("model_version"),
        "status": data.get("status", "ok"),
    }


@router.get("/recommendations")
async def get_recommendations(request: Request, top_k: int = 10):
    """
    Get recommendations for the logged-in user.

    Requires recommender pipeline to have been run
    (ingest data, then: python -m recommender.pipeline --config configs/base.yaml).
    """
    username = get_current_username(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        from core.config import get_project_root, load_config

        root = get_project_root()
        cfg = load_config()
        artifacts_dir = Path(
            cfg.get("paths", {}).get("artifacts", str(root / "artifacts"))
        )
        if not (artifacts_dir / "als_model.pkl").exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Recommender not trained. Ingest data then run: "
                    "python -m recommender.pipeline --config "
                    "configs/base.yaml"
                ),
            )
        from recommender.pipeline import recommend, load_pipeline_artifacts

        pipeline_artifacts = load_pipeline_artifacts(artifacts_dir)
        if not pipeline_artifacts:
            raise HTTPException(
                status_code=503,
                detail="Could not load recommender artifacts",
            )
        out = recommend(
            username, pipeline_artifacts, top_k=top_k, exclude_owned=True
        )
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ConditionRequest(BaseModel):
    seller_notes: str
    item_id: str | None = None
    metadata: dict | None = None


@lru_cache(maxsize=1)
def _get_baseline_pipeline():
    """
    Lazily instantiate the grader inference pipeline once per process.

    Baseline is the default for personal/offline use:
    TF-IDF + Logistic Regression tends to be smaller and faster than
    the transformer.
    """
    from grader.src.pipeline import Pipeline

    from core.config import get_project_root

    root = get_project_root()
    config_path = root / "grader" / "configs" / "grader.yaml"
    guidelines_path = root / "grader" / "configs" / "grading_guidelines.yaml"

    pl = Pipeline(
        config_path=str(config_path),
        guidelines_path=str(guidelines_path),
    )
    pl.infer_model = "baseline"
    return pl


@router.post("/condition")
async def predict_condition(payload: ConditionRequest):
    """
    Predict sleeve and media condition from seller notes (vinyl condition grader).
    """
    try:
        seller_notes = (payload.seller_notes or "").strip()
        if not seller_notes:
            raise HTTPException(
                status_code=400, detail="seller_notes is required"
            )

        pl = _get_baseline_pipeline()
        return pl.predict(
            text=seller_notes,
            item_id=payload.item_id,
            metadata=payload.metadata,
        )
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if any(s in lower for s in _BASELINE_ARTIFACT_ERROR_HINTS):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Grader baseline artifacts not available. "
                    "Train/export the grader model first."
                ),
            )
        raise HTTPException(status_code=500, detail=msg)


@router.get("/price/{release_id}")
async def estimate_price(
    release_id: str,
    sleeve_condition: str | None = None,
    media_condition: str | None = None,
):
    """
    Price estimate for a Discogs release.

    If ``PRICE_SERVICE_URL`` is set (VinylIQ microservice), proxies to
    ``POST {PRICE_SERVICE_URL}/estimate``. Otherwise calls in-process
    ``price_estimator.src.pipeline.estimate``.
    """
    base = (os.environ.get("PRICE_SERVICE_URL") or "").strip().rstrip("/")
    if base:
        try:
            import httpx

            headers = {}
            key = os.environ.get("VINYLIQ_API_KEY")
            if key:
                headers["X-API-Key"] = key
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{base}/estimate",
                    json={
                        "release_id": release_id,
                        "sleeve_condition": sleeve_condition,
                        "media_condition": media_condition,
                        "refresh_stats": False,
                    },
                    headers=headers,
                )
            if r.status_code >= 400:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=r.text or r.reason_phrase,
                )
            data = r.json()
            return _web_price_response_from_service(
                data, release_id, sleeve_condition, media_condition
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
    try:
        from price_estimator.src.pipeline import estimate

        return estimate(
            release_id,
            sleeve_condition=sleeve_condition,
            media_condition=media_condition,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
