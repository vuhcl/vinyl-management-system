"""Grader in-process schema types (not HTTP / MLflow wire formats)."""

from __future__ import annotations

from grader.src.schemas.description_merge import (
    DESCRIPTION_QUALITY_META_KEYS,
    merge_description_quality_metadata,
)
from grader.src.schemas.prediction import (
    ConfidenceScoresBundle,
    GraderPrediction,
    PerTargetGradeProbs,
    PredictionMetadata,
)

__all__ = [
    "DESCRIPTION_QUALITY_META_KEYS",
    "ConfidenceScoresBundle",
    "GraderPrediction",
    "PerTargetGradeProbs",
    "PredictionMetadata",
    "merge_description_quality_metadata",
]
