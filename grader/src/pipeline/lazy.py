"""Lazy initialization of preprocessor, rule engine, TF-IDF, and model slots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from grader.src.config_io import load_yaml_mapping
from grader.src.mlflow_tracking import configure_mlflow_from_config, mlflow_enabled
from grader.src.data.preprocess import Preprocessor
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer
from grader.src.rules.rule_engine import RuleEngine


class PipelineLazyMixin:
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
