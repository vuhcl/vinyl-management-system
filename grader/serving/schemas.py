"""
Pydantic request/response models for the grader serving API.

``PredictRequest`` accepts either a single ``text`` or a batch ``items`` list
(see ``MAX_BATCH`` / ``MAX_TEXT_LEN``). The FastAPI handler builds a dataframe
from whichever mode was used.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator


MAX_BATCH = 256
MAX_TEXT_LEN = 16_000


class PredictItem(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    item_id: Optional[Union[str, int]] = None


class PredictRequest(BaseModel):
    """Exactly one of: non-empty ``text`` or non-empty ``items``."""

    text: Optional[str] = Field(None, max_length=MAX_TEXT_LEN)
    items: Optional[list[PredictItem]] = Field(None, max_length=MAX_BATCH)

    @model_validator(mode="after")
    def exactly_one_mode(self) -> "PredictRequest":
        has_text = self.text is not None and self.text.strip() != ""
        has_items = bool(self.items)
        if has_text == has_items:
            raise ValueError(
                "Provide exactly one of: non-empty 'text' or non-empty 'items'"
            )
        return self


class PredictionRow(BaseModel):
    item_id: Any
    predicted_sleeve_condition: str
    predicted_media_condition: str
    sleeve_confidence: float
    media_confidence: float
    # Post-rule flags (grades above reflect RuleEngine output).
    # Confidences remain the pyfunc model top-1 scores (see API docstring).
    contradiction_detected: bool = False
    rule_override_applied: bool = False
    rule_override_target: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: list[PredictionRow]
