"""
Structured types for in-process grader model outputs (before/after RuleEngine).

FIELD INVENTORY
---------------
**Top-level keys (all producers):**

- ``item_id`` — often ``str``; may be ``int`` or other JSON-serializable
  values from callers / pyfunc rows (see :class:`GraderPrediction`).
- ``predicted_sleeve_condition``, ``predicted_media_condition`` —
  ``str`` grade labels.
- ``confidence_scores`` — :class:`ConfidenceScoresBundle`: full per-class
  softmax maps from baseline/transformer, or degenerate singleton maps from
  MLflow pyfunc adapter (one probability per target for the predicted grade;
  sufficient for RuleEngine).
- ``metadata`` — ``dict[str, Any]``; **must exist** before
  :meth:`RuleEngine.apply` (models + serving adapters supply it).

**Baseline / transformer model metadata** (subset varies by path):

- ``source``, ``media_verifiable``, ``media_evidence_strength``,
  ``confidence_band_*``, ``excellent_proxy_media``,
  ``media_evidence_scores``, ``ambiguous_prediction``,
  ``rule_override_applied``, ``rule_override_target``,
  ``contradiction_detected``.

**Merged from preprocess records** (see
:func:`grader.src.schemas.description_merge.merge_description_quality_metadata`):

- ``sleeve_note_adequate``, ``media_note_adequate``,
  ``adequate_for_training``, ``needs_richer_note``,
  ``description_quality_gaps``, ``description_quality_prompts``.

**Rule engine writes** (see ``rule_engine/application.py``):

- ``contradiction_detected``, ``rule_override_applied``,
  ``rule_override_target``, optionally ``excellent_collapsed_to_near_mint``,
  ``hard_override_tier_*``, ``hard_override_signal_*``.

**Consumers that index prediction-shaped dicts** (for audits / tooling):

- ``grader/src/eval/export_mispredictions.py``
- ``grader/src/eval/analyze_harmful_overrides.py``
- ``grader/src/eval/analyze_missed_rule_owned.py``
- ``grader/src/evaluation/rule_engine_eval.py``
  (via pipeline rule-eval path)

Serialized shapes are **not** part of this module: MLflow pyfunc DataFrame
columns (``grader_pyfunc``) and FastAPI ``PredictionRow`` remain stable by
design.
"""

from __future__ import annotations

from typing import Any, TypedDict

# Per sleeve or per media: full softmax or pyfunc singleton
# ``{predicted_grade: p}``
PerTargetGradeProbs = dict[str, float]


class PredictionMetadata(TypedDict, total=False):
    """
    Documented metadata keys for predictions (subset may be present).

    Runtime ``metadata`` may include additional keys; keep ``dict[str, Any]`` on
    :class:`GraderPrediction` for arbitrary extensions (rubrics, experiments).
    """

    source: str
    media_verifiable: bool
    media_evidence_strength: str
    confidence_band_sleeve: Any
    confidence_band_media: Any
    excellent_proxy_media: float
    media_evidence_scores: dict[str, float]
    ambiguous_prediction: bool
    contradiction_detected: bool
    rule_override_applied: bool
    rule_override_target: Any
    excellent_collapsed_to_near_mint: list[str]
    hard_override_tier_sleeve: Any
    hard_override_tier_media: Any
    hard_override_signal_sleeve: Any
    hard_override_signal_media: Any
    sleeve_note_adequate: bool
    media_note_adequate: bool
    adequate_for_training: bool
    needs_richer_note: bool
    description_quality_gaps: Any
    description_quality_prompts: Any


class ConfidenceScoresBundle(TypedDict):
    """Sleeve and media grade → probability maps (full or top-1-only)."""

    sleeve: PerTargetGradeProbs
    media: PerTargetGradeProbs


class GraderPrediction(TypedDict):
    """
    In-process prediction dict passed through models, pipeline, and RuleEngine.

    ``item_id`` is loose: DataFrame / API rows may use non-``str`` ids; use
    :func:`typing.cast` at JSON or pandas boundaries if a checker complains.
    """

    item_id: Any
    predicted_sleeve_condition: str
    predicted_media_condition: str
    confidence_scores: ConfidenceScoresBundle
    metadata: dict[str, Any]
