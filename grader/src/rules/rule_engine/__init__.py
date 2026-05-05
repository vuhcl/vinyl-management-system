"""
grader/src/rules/rule_engine/

Post-processing rule engine for vinyl condition grading.
Applies grading rubric rules on top of model predictions.

Rule priority (strictly enforced):
  1. Contradiction detection — flag and suppress all overrides
  2. Hard signal overrides   — Poor, Generic (unconditional). On **sleeve**,
     ``check_hard_override`` evaluates **Poor before Generic** so catastrophic
     jacket damage outranks generic-housing cues (e.g. a destroyed jacket
     shipped in a white generic sleeve). Media evaluates only Poor among
     hard-owned grades (Generic is sleeve-only).
  3. Soft signal overrides   — other grades (confidence-gated)

Operates independently on sleeve and media targets.
Generic checks are skipped for media — Generic is sleeve-only.

Input:  :class:`grader.src.schemas.GraderPrediction` from baseline/transformer models
        or the MLflow pyfunc adapter (serving); plain dicts that match that shape are
        accepted at runtime.
Output: same schema with updated grade and metadata fields

Usage:
    from grader.src.rules.rule_engine import RuleEngine

    engine = RuleEngine(guidelines_path="grader/configs/grading_guidelines.yaml")
    prediction = engine.apply(prediction, text)
    predictions = engine.apply_batch(predictions, texts)
"""

from grader.src.rules.rule_engine.engine import RuleEngine

__all__ = ["RuleEngine"]
