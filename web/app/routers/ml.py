"""
ML component APIs: recommender, condition classifier, price estimator.
These delegate to the respective subproject pipelines/artifacts.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from core.auth import get_user_token

router = APIRouter()


def _username(request: Request) -> str | None:
    return getattr(request.state, "username", None)


@router.get("/recommendations")
async def get_recommendations(request: Request, top_k: int = 10):
    """
    Get recommendations for the logged-in user. Requires recommender pipeline to have been run
    (ingest data, then: python -m recommender.pipeline --config configs/base.yaml).
    """
    username = _username(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        from core.config import get_project_root, load_config
        root = get_project_root()
        cfg = load_config()
        artifacts_dir = Path(cfg.get("paths", {}).get("artifacts", str(root / "artifacts")))
        if not (artifacts_dir / "als_model.pkl").exists():
            raise HTTPException(
                status_code=503,
                detail="Recommender not trained. Ingest data then run: python -m recommender.pipeline --config configs/base.yaml",
            )
        # Load saved artifacts and call recommend (recommender exposes recommend(user_id, artifacts, top_k))
        from recommender.pipeline import recommend, load_pipeline_artifacts
        pipeline_artifacts = load_pipeline_artifacts(artifacts_dir)
        if not pipeline_artifacts:
            raise HTTPException(status_code=503, detail="Could not load recommender artifacts")
        out = recommend(username, pipeline_artifacts, top_k=top_k, exclude_owned=True)
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/condition")
async def predict_condition(seller_notes: str):
    """
    Predict sleeve and media condition from seller notes (NLP classifier).
    """
    try:
        # Stub: call nlp_condition_classifier if artifact available
        return {
            "seller_notes": seller_notes[:200],
            "predicted_sleeve_condition": None,
            "predicted_media_condition": None,
            "status": "stub",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{release_id}")
async def estimate_price(release_id: str, sleeve_condition: str | None = None, media_condition: str | None = None):
    """
    Price estimate for a Discogs release (optionally with condition from NLP).
    """
    try:
        from price_estimator.src.pipeline import estimate
        return estimate(release_id, sleeve_condition=sleeve_condition, media_condition=media_condition)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
