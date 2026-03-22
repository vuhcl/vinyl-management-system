"""
ML component APIs: recommender, condition classifier, price estimator.
These delegate to the respective subproject pipelines/artifacts.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from functools import lru_cache
from pydantic import BaseModel

router = APIRouter()


def _username(request: Request) -> str | None:
    return getattr(request.state, "username", None)


@router.get("/recommendations")
async def get_recommendations(request: Request, top_k: int = 10):
    """
    Get recommendations for the logged-in user.

    Requires recommender pipeline to have been run
    (ingest data, then: python -m recommender.pipeline --config configs/base.yaml).
    """
    username = _username(request)
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
        # Load saved artifacts and call recommend (recommender exposes recommend(user_id, artifacts, top_k))
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
    # Optional fields for callers (mobile app can omit).
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
        if any(
            s in msg.lower()
            for s in [
                "no such file",
                "not found",
                "model not calibrated",
                "pickle",
                "pkl",
                "vectorizer",
                "encoder",
                "features",
            ]
        ):
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
    Price estimate for a Discogs release (optionally with condition from NLP).
    """
    try:
        from price_estimator.src.pipeline import estimate
        return estimate(
            release_id,
            sleeve_condition=sleeve_condition,
            media_condition=media_condition,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
