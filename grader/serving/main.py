"""FastAPI app: load grader from MLflow once at startup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from fastapi import FastAPI, HTTPException

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


def _pkg_version() -> str:
    try:
        return version("vinyl-management-system")
    except PackageNotFoundError:
        return "0.0.0-dev"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_grader_pyfunc()
    logger.info("Grader pyfunc model loaded successfully")
    init_rule_stack()
    verify_serving_guidelines_pairing()
    yield
    _model = None


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


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    from grader.src.guidelines_identity import guidelines_version_from_mapping

    try:
        from grader.serving.rule_postprocess import get_rule_engine

        gv = guidelines_version_from_mapping(get_rule_engine().guidelines)
    except RuntimeError:
        gv = None
    return {
        "status": "ok",
        "model_loaded": True,
        "guidelines_version": gv,
        "model_guidelines_version_tag": get_model_guidelines_version_tag(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    if body.text is not None and body.text.strip():
        df = pd.DataFrame([{"text": body.text.strip(), "item_id": 0}])
    else:
        assert body.items is not None
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

    predictions = []
    for pred, (_, prow) in zip(final, out.iterrows()):
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
                sleeve_confidence=float(prow["sleeve_confidence"]),
                media_confidence=float(prow["media_confidence"]),
                contradiction_detected=bool(
                    meta.get("contradiction_detected", False)
                ),
                rule_override_applied=bool(
                    meta.get("rule_override_applied", False)
                ),
                rule_override_target=meta.get("rule_override_target"),
                guidelines_version=meta.get("guidelines_version"),
                model_guidelines_version=get_model_guidelines_version_tag(),
            )
        )
    return PredictResponse(predictions=predictions)
