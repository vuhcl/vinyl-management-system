"""Lazy initialization, registry hook, and composed :class:`Pipeline`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow

from grader.src.config_io import load_yaml_mapping
from grader.src.data.preprocess import Preprocessor
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    vinyl_grader_pyfunc_has_python_model,
)
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer
from grader.src.rules.rule_engine import RuleEngine

from .inference import _PipelineInference
from .training import _PipelineTraining

logger = logging.getLogger(__name__)


class _PipelineLazy:
    """Config wiring and lazy-loaded inference components."""

    config_path: str
    guidelines_path: str
    config: dict
    infer_model: str
    artifacts_dir: Path
    _preprocessor: Optional[Preprocessor]
    _rule_engine: Optional[RuleEngine]
    _baseline: Optional[BaselineModel]
    _transformer: Optional[TransformerTrainer]
    _tfidf: Optional[TFIDFFeatureBuilder]

    def __init__(
        self,
        config_path: str = "grader/configs/grader.yaml",
        guidelines_path: str = "grader/configs/grading_guidelines.yaml",
    ) -> None:
        self.config_path = config_path
        self.guidelines_path = guidelines_path
        self.config = load_yaml_mapping(config_path)

        inference_cfg = self.config.get("inference", {})
        self.infer_model = inference_cfg.get("model", "transformer")

        self.artifacts_dir = Path(self.config["paths"]["artifacts"])

        if mlflow_enabled(self.config):
            configure_mlflow_from_config(self.config)

        self._preprocessor: Optional[Preprocessor] = None
        self._rule_engine: Optional[RuleEngine] = None
        self._baseline: Optional[BaselineModel] = None
        self._transformer: Optional[TransformerTrainer] = None
        self._tfidf: Optional[TFIDFFeatureBuilder] = None

    def _get_preprocessor(self) -> Preprocessor:
        if self._preprocessor is None:
            self._preprocessor = Preprocessor(
                config_path=self.config_path,
                guidelines_path=self.guidelines_path,
            )
        return self._preprocessor

    def _get_rule_engine(self) -> RuleEngine:
        if self._rule_engine is None:
            rules_cfg = self.config.get("rules") or {}
            allow_ex = bool(
                rules_cfg.get("allow_excellent_soft_override", False)
            )
            self._rule_engine = RuleEngine(
                guidelines_path=self.guidelines_path,
                allow_excellent_soft_override=allow_ex,
            )
        return self._rule_engine

    def _get_tfidf(self) -> TFIDFFeatureBuilder:
        if self._tfidf is None:
            self._tfidf = TFIDFFeatureBuilder(
                config_path=self.config_path
            )
        return self._tfidf


class _PipelineRegistry:
    """Register transformer pyfunc to MLflow model registry when configured."""

    config: dict

    def _register_transformer_to_registry(
        self,
        transformer_results: dict,
        *,
        want_register: bool,
        registry_model_name: str,
    ) -> None:
        if not want_register:
            return
        if not mlflow_log_artifacts_enabled(self.config):
            logger.info(
                "Skipping MLflow model registry (mlflow.log_artifacts: false)."
            )
            return
        run_id = (transformer_results or {}).get("mlflow_run_id") or ""
        if not run_id:
            logger.warning(
                "Skipping MLflow model registry: empty mlflow_run_id "
                "(enable MLflow for the transformer step to register)."
            )
            return
        configure_mlflow_from_config(self.config)
        if not vinyl_grader_pyfunc_has_python_model(run_id):
            logger.warning(
                "Skipping MLflow model registry: run %s has no usable "
                "vinyl_grader pyfunc (no python_model.pkl and no models-from-code "
                "entry in MLmodel). Remote pyfunc logging may have failed "
                "(see training logs for 'pyfunc logging failed'); increase "
                "MLFLOW_HTTP_REQUEST_TIMEOUT and related upload settings — "
                "grader/serving/README.md.",
                run_id,
            )
            return
        model_uri = f"runs:/{run_id}/vinyl_grader"
        try:
            ev = transformer_results.get("eval", {}).get("test", {})
            s_f1 = float(ev.get("sleeve", {}).get("macro_f1", 0.0))
            m_f1 = float(ev.get("media", {}).get("macro_f1", 0.0))
            mean_f1 = (s_f1 + m_f1) / 2.0
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=registry_model_name,
                tags={
                    "source": "full_pipeline",
                    "test_mean_macro_f1": str(round(mean_f1, 4)),
                },
            )
            logger.info(
                "Registered model '%s' version %s from run %s (%s)",
                registry_model_name,
                mv.version,
                run_id,
                model_uri,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model registration failed: %s", exc)


class Pipeline(
    _PipelineLazy,
    _PipelineRegistry,
    _PipelineTraining,
    _PipelineInference,
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
