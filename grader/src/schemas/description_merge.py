"""Merge preprocess description-quality fields into prediction metadata."""

from __future__ import annotations

from grader.src.schemas.prediction import GraderPrediction

# Keys copied from preprocess ``records`` into each prediction's ``metadata``.
DESCRIPTION_QUALITY_META_KEYS: tuple[str, ...] = (
    "sleeve_note_adequate",
    "media_note_adequate",
    "adequate_for_training",
    "needs_richer_note",
    "description_quality_gaps",
    "description_quality_prompts",
)


def merge_description_quality_metadata(
    predictions: list[GraderPrediction],
    records: list[dict],
) -> None:
    """Copy note-adequacy fields from records into prediction metadata."""
    for pred, rec in zip(predictions, records):
        meta = pred.setdefault("metadata", {})
        for k in DESCRIPTION_QUALITY_META_KEYS:
            if k in rec:
                meta[k] = rec[k]
