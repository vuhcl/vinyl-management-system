"""Composed :class:`Pipeline` — top-level grader train/infer orchestrator."""

from __future__ import annotations

from .inference import PipelineInferenceMixin
from .lazy import PipelineLazyMixin
from .registry import PipelineRegistryMixin
from .rule_eval import PipelineRuleEvalMixin
from .training import PipelineTrainingMixin


class Pipeline(
    PipelineLazyMixin,
    PipelineRegistryMixin,
    PipelineTrainingMixin,
    PipelineInferenceMixin,
    PipelineRuleEvalMixin,
):
    """
    Top-level orchestrator for training and inference.

    Training pipeline:
        Runs the full sequence from ingestion to model comparison.
        Each step is individually skippable for development iterations.

    Inference pipeline:
        Loads pre-trained artifacts and runs single or batch prediction.
        All preprocessing is handled internally — callers pass raw text.

    Config keys read from grader.yaml:
        inference.model             — "baseline" or "transformer"
        paths.*                     — all artifact and data paths
        mlflow.*                    — experiment tracking config
    """
