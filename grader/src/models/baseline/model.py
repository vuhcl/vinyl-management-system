"""Concrete `BaselineModel` class composed from feature/train/infer/... mixins."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from grader.src.config_io import load_yaml_mapping
from grader.src.mlflow_tracking import configure_mlflow_from_config, mlflow_enabled

from .artifacts import BaselineArtifactsMixin
from .constants import TARGETS
from .evaluation import BaselineEvaluationMixin
from .features import BaselineFeaturesMixin
from .inference import BaselineInferenceMixin
from .isotonic import _IsotonicCalibrator
from .orchestrate import BaselineOrchestrationMixin
from .train import BaselineTrainMixin


class BaselineModel(
    BaselineFeaturesMixin,
    BaselineTrainMixin,
    BaselineInferenceMixin,
    BaselineEvaluationMixin,
    BaselineArtifactsMixin,
    BaselineOrchestrationMixin,
):
    """
    Two-head logistic regression classifier for vinyl condition grading.

    Each head is an independent LogisticRegression fitted on TF-IDF
    features for a single target (sleeve or media). Probability outputs
    are calibrated using isotonic regression on the val split.

    Config keys read from grader.yaml:
        models.baseline.logistic_regression.*  — LR hyperparameters
        evaluation.calibration.method          — calibration method
        paths.artifacts                         — artifact directory
        mlflow (URI from MLFLOW_TRACKING_URI / tracking_uri_fallback)
        mlflow.experiment_name
    """

    def __init__(
        self,
        config_path: str,
        config: Optional[dict] = None,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = load_yaml_mapping(config_path)
        self._config_path_str = config_path

        lr_cfg = self.config["models"]["baseline"]["logistic_regression"]
        self.C: float = lr_cfg["C"]
        self.max_iter: int = lr_cfg["max_iter"]
        self.class_weight: str = lr_cfg["class_weight"]
        self.solver: str = lr_cfg["solver"]
        self.random_state: int = lr_cfg.get("random_state", 42)
        self.tol: float = float(lr_cfg.get("tol", 1e-4))
        tuning_cfg = self.config["models"]["baseline"].get("tuning", {})
        self.tuning_enabled: bool = bool(tuning_cfg.get("enabled", False))
        self.tuning_c_values: list[float] = [
            float(v) for v in tuning_cfg.get("c_values", [self.C])
        ]
        boundary_cfg = self.config["models"]["baseline"].get(
            "boundary_objective", {}
        )
        self.boundary_enabled: bool = bool(boundary_cfg.get("enabled", False))
        self.boundary_alpha: float = float(boundary_cfg.get("alpha", 0.2))
        band_cfg = self.config["models"]["baseline"].get("confidence_band", {})
        self.confidence_band_enabled: bool = bool(band_cfg.get("enabled", True))
        self.confidence_band_max_gap: float = float(
            band_cfg.get("max_gap", 0.12)
        )
        gating_cfg = self.config["models"]["baseline"].get(
            "media_evidence_gating", {}
        )
        self.media_gating_enabled: bool = bool(gating_cfg.get("enabled", False))
        self.media_gating_alpha_none: float = float(
            gating_cfg.get("alpha_none", 0.35)
        )
        self.media_gating_alpha_weak: float = float(
            gating_cfg.get("alpha_weak", 0.20)
        )
        aux_cfg = self.config["models"]["baseline"].get(
            "media_evidence_aux", {}
        )
        self.media_evidence_aux_enabled: bool = bool(
            aux_cfg.get("enabled", False)
        )

        cal_cfg = self.config["evaluation"]["calibration"]
        self.calibration_method: str = cal_cfg.get("method", "isotonic")

        self.artifacts_dir = Path(self.config["paths"]["artifacts"])
        self.features_dir = self.artifacts_dir / "features"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.model_paths = {
            target: self.artifacts_dir / f"baseline_{target}.pkl"
            for target in TARGETS
        }
        self.calibrated_paths = {
            target: self.artifacts_dir / f"baseline_{target}_calibrated.pkl"
            for target in TARGETS
        }
        self.confusion_paths = {
            target: self.artifacts_dir / f"confusion_matrix_{target}.txt"
            for target in TARGETS
        }

        self.models: dict[str, LogisticRegression] = {}
        self.calibrated: dict[str, _IsotonicCalibrator] = {}
        self.boundary_models: dict[str, LogisticRegression] = {}
        self.boundary_band_labels: dict[str, np.ndarray] = {}
        self.media_evidence_aux_model: Optional[LogisticRegression] = None
        self.encoders: dict = {}
        self.selected_c: dict[str, float] = {}
        self.grade_ordinal_map = self._load_grade_ordinal_map()

        if mlflow_enabled(self.config):
            configure_mlflow_from_config(self.config)
