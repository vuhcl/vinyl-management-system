"""FastAPI app: load grader from MLflow once at startup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from grader.serving.guidelines_pairing import (
    get_model_guidelines_version_tag,
    verify_serving_guidelines_pairing,
)
from grader.serving.model_loader import load_grader_pyfunc
from grader.serving.rule_postprocess import (
    apply_rules_to_pyfunc_batch,
    init_rule_stack,
)
from grader.serving.schemas import (
    MAX_BATCH,
    PredictRequest,
    PredictResponse,
    PredictionRow,
)

logger = logging.getLogger(__name__)

_model = None
_health_snapshot: dict[str, Any] | None = None


def _pkg_version() -> str:
    try:
        return version("vinyl-management-system")
    except PackageNotFoundError:
        return "0.0.0-dev"


def _build_health_snapshot() -> dict[str, Any]:
    from grader.serving.rule_postprocess import get_rule_engine
    from grader.src.guidelines_identity import guidelines_version_from_mapping

    gv = guidelines_version_from_mapping(get_rule_engine().guidelines)
    return {
        "status": "ok",
        "model_loaded": True,
        "guidelines_version": gv,
        "model_guidelines_version_tag": get_model_guidelines_version_tag(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _health_snapshot
    _model = load_grader_pyfunc()
    logger.info("Grader pyfunc model loaded successfully")
    init_rule_stack()
    verify_serving_guidelines_pairing()
    try:
        _health_snapshot = _build_health_snapshot()
    except RuntimeError as exc:
        logger.warning("Health snapshot unavailable at startup: %s", exc)
        _health_snapshot = {
            "status": "ok",
            "model_loaded": True,
            "guidelines_version": None,
            "model_guidelines_version_tag": get_model_guidelines_version_tag(),
        }
    yield
    _model = None
    _health_snapshot = None


app = FastAPI(
    title="Vinyl grader",
    description=(
        "Condition grading: MLflow DistilBERT pyfunc, then the same "
        "Preprocessor + RuleEngine as grader.src.pipeline inference. "
        "Sleeve/media confidences are top-1 softmax probabilities "
        "(optional GRADER_SOFTMAX_TEMPERATURE / GRADER_*_SOFTMAX_TEMPERATURE "
        "env vars flatten overconfident peaks); unchanged when rules adjust a grade."
    ),
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "vinyl-grader",
        "version": _pkg_version(),
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health(request: Request):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    body = _health_snapshot
    if body is None:
        try:
            body = _build_health_snapshot()
        except RuntimeError as exc:
            logger.warning("Health snapshot rebuild failed: %s", exc)
            body = {
                "status": "ok",
                "model_loaded": True,
                "guidelines_version": None,
                "model_guidelines_version_tag": get_model_guidelines_version_tag(),
            }
    if request.method == "HEAD":
        return Response(status_code=200)
    return body


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    if body.text is not None and body.text.strip():
        df = pd.DataFrame([{"text": body.text.strip(), "item_id": 0}])
    else:
        if body.items is None:
            raise HTTPException(
                status_code=422,
                detail="Provide non-empty 'items' when 'text' is omitted",
            )
        rows = []
        for i, it in enumerate(body.items):
            rows.append(
                {
                    "text": it.text.strip(),
                    "item_id": it.item_id if it.item_id is not None else i,
                }
            )
        df = pd.DataFrame(rows)

    if len(df) > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"batch size {len(df)} exceeds max {MAX_BATCH}",
        )

    out = _model.predict(df)
    raw_texts = df["text"].astype(str).tolist()
    item_ids = df["item_id"].tolist()
    metadata_list = [{} for _ in range(len(df))]

    final = apply_rules_to_pyfunc_batch(
        out, raw_texts, item_ids, metadata_list
    )

    model_gv = get_model_guidelines_version_tag()
    sleeve_confidences = out["sleeve_confidence"].tolist()
    media_confidences = out["media_confidence"].tolist()

    predictions = []
    for pred, sleeve_confidence, media_confidence in zip(
        final, sleeve_confidences, media_confidences
    ):
        meta = pred["metadata"]
        predictions.append(
            PredictionRow(
                item_id=pred["item_id"],
                predicted_sleeve_condition=str(
                    pred["predicted_sleeve_condition"]
                ),
                predicted_media_condition=str(
                    pred["predicted_media_condition"]
                ),
                sleeve_confidence=float(sleeve_confidence),
                media_confidence=float(media_confidence),
                contradiction_detected=bool(
                    meta.get("contradiction_detected", False)
                ),
                rule_override_applied=bool(
                    meta.get("rule_override_applied", False)
                ),
                rule_override_target=meta.get("rule_override_target"),
                guidelines_version=meta.get("guidelines_version"),
                model_guidelines_version=model_gv,
            )
        )
    return PredictResponse(predictions=predictions)
