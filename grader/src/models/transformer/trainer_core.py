"""TransformerTrainer: composed class with config wiring."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import DistilBertTokenizerFast

from grader.src.config_io import load_yaml_mapping
from grader.src.mlflow_tracking import configure_mlflow_for_transformer_init

from .constants import SPLITS
from .device import get_device
from .hparams import merge_transformer_hparams, resolve_model_artifact_dir
from .trainer_data import TransformerDataMixin
from .trainer_predict_io import TransformerPredictIoMixin
from .trainer_run import TransformerRunMixin
from .trainer_training import TransformerTrainingMixin
from .two_head import TwoHeadClassifier


class TransformerTrainer(
    TransformerDataMixin,
    TransformerTrainingMixin,
    TransformerPredictIoMixin,
    TransformerRunMixin,
):
    """
    Training orchestrator for the two-head DistilBERT classifier.

    Handles data loading, training loop, early stopping, evaluation,
    model serialization, and MLflow logging.

    Config keys read from grader.yaml:
        models.transformer.*        — all model and training hyperparameters
        paths.splits                — JSONL split files
        paths.artifacts             — output directory
        mlflow.tracking_uri / experiment_name
            (configure_mlflow_from_config: env MLFLOW_TRACKING_URI
            overrides mlflow.tracking_uri_fallback in grader.yaml)
    """

    def __init__(
        self,
        config_path: str,
        freeze_encoder_override: Optional[bool] = None,
        transformer_overrides: Optional[dict[str, Any]] = None,
        artifact_subdir: Optional[str] = None,
        config: Optional[dict] = None,
        *,
        tuning: bool = False,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = load_yaml_mapping(config_path)
        if transformer_overrides:
            self.config["models"]["transformer"] = merge_transformer_hparams(
                self.config["models"]["transformer"],
                transformer_overrides,
            )
        self._mlflow_tuning: bool = tuning

        t_cfg = self.config["models"]["transformer"]
        self.base_model: str = t_cfg["base_model"]
        self.freeze_encoder: bool = (
            freeze_encoder_override
            if freeze_encoder_override is not None
            else t_cfg["freeze_encoder"]
        )
        self.max_length: int = t_cfg["max_length"]
        self.dropout: float = t_cfg["dropout"]
        self.lr: float = t_cfg["learning_rate"]
        self.batch_size: int = t_cfg["batch_size"]
        self.epochs: int = t_cfg["epochs"]
        self.patience: int = t_cfg["early_stopping_patience"]
        self.class_weight: str = t_cfg["class_weight"]
        self.random_state: int = t_cfg.get("random_state", 42)
        self.weight_decay: float = float(t_cfg.get("weight_decay", 0.01))
        self.warmup_ratio: float = float(t_cfg.get("warmup_ratio", 0.1))
        self.unfreeze_top_n_layers: int = int(
            t_cfg.get("unfreeze_top_n_layers", 0)
        )
        mea = t_cfg.get("media_evidence_aux") or {}
        self.media_evidence_aux_enabled: bool = bool(mea.get("enabled", False))
        self.media_evidence_aux_weight: float = float(
            mea.get("loss_weight", 0.25)
        )

        splits_dir = Path(self.config["paths"]["splits"])
        self.artifacts_dir = Path(self.config["paths"]["artifacts"])
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_artifact_dir = resolve_model_artifact_dir(
            self.artifacts_dir,
            artifact_subdir,
        )
        self.model_artifact_dir.mkdir(parents=True, exist_ok=True)

        self.split_paths = {
            split: splits_dir / f"{split}.jsonl" for split in SPLITS
        }

        self.weights_path = self.model_artifact_dir / "transformer_weights.pt"
        self.config_path_out = (
            self.model_artifact_dir / "transformer_config.json"
        )
        self.tokenizer_dir = self.model_artifact_dir / "tokenizer"

        self.device = get_device()
        self._skip_mlflow: bool = False

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        configure_mlflow_for_transformer_init(
            self.config, tuning=self._mlflow_tuning
        )

        self.model: Optional[TwoHeadClassifier] = None
        self.tokenizer: Optional[DistilBertTokenizerFast] = None
        self.encoders: dict = {}
