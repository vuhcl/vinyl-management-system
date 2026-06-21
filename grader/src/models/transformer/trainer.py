"""TransformerTrainer: composed class with config wiring."""

from __future__ import annotations

import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

from grader.src.config_io import load_yaml_mapping
from grader.src.mlflow_tracking import configure_mlflow_for_transformer_init

from .constants import SPLITS, TARGETS
from .device import get_device
from .hparams import merge_transformer_hparams, resolve_model_artifact_dir
from .trainer_run import _TransformerPredictIo, _TransformerRun
from .two_head import TwoHeadClassifier

logger = logging.getLogger(__name__)


class _TransformerData:
    """Split file IO and encoder pickle loading."""

    artifacts_dir: Path
    split_paths: dict

    def _load_jsonl(self, path: Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d records from %s", len(records), path.name)
        return records

    def load_encoders(self) -> dict:
        """
        Load label encoders fitted by tfidf_features.py.
        Shared encoders ensure class indices are identical across
        baseline and transformer for valid metric comparison.
        """
        encoders = {}
        for target in TARGETS:
            path = self.artifacts_dir / f"label_encoder_{target}.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Label encoder not found: {path}. "
                    "Run tfidf_features.py first."
                )
            with open(path, "rb") as f:
                encoders[target] = pickle.load(f)
            logger.info(
                "Loaded label encoder — target=%s classes=%s",
                target,
                list(encoders[target].classes_),
            )
        return encoders


class _TransformerTraining:
    """train_epoch / eval_epoch / train inner loop."""

    model: nn.Module
    device: torch.device
    lr: float
    weight_decay: float
    warmup_ratio: float
    epochs: int
    patience: int
    batch_size: int
    media_evidence_aux_weight: float
    media_evidence_aux_enabled: bool
    class_weight: str
    _skip_mlflow: bool

    def _compute_class_weights(
        self,
        records: list[dict],
        target: str,
        encoder,
    ) -> Optional[torch.Tensor]:
        """
        Compute inverse-frequency class weights for CrossEntropyLoss.
        Only applied when class_weight == "balanced" in config.
        """
        if self.class_weight != "balanced":
            return None

        labels = encoder.transform([r[f"{target}_label"] for r in records])
        n_classes = len(encoder.classes_)
        counts = np.bincount(labels, minlength=n_classes).astype(float)

        counts = np.maximum(counts, 1.0)
        weights = len(labels) / (n_classes * counts)
        weights = weights / weights.sum() * n_classes

        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer,
        scheduler,
        sleeve_criterion: nn.CrossEntropyLoss,
        media_criterion: nn.CrossEntropyLoss,
        evidence_criterion: Optional[nn.CrossEntropyLoss] = None,
    ) -> float:
        """
        Run one training epoch.

        Returns:
            Mean combined loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sleeve_labels = batch["sleeve_label"].to(self.device)
            media_labels = batch["media_label"].to(self.device)

            optimizer.zero_grad()

            try:
                _out = self.model(input_ids, attention_mask)
            except RuntimeError as e:
                if "MPS" in str(e) or "not currently supported" in str(e):
                    logger.warning(
                        "MPS op not supported — falling back to CPU. "
                        "Error: %s",
                        e,
                    )
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    sleeve_labels = sleeve_labels.to(self.device)
                    media_labels = media_labels.to(self.device)
                    _out = self.model(input_ids, attention_mask)
                else:
                    raise

            if len(_out) == 3:
                sleeve_logits, media_logits, ev_logits = _out
            else:
                sleeve_logits, media_logits = _out
                ev_logits = None

            loss = sleeve_criterion(
                sleeve_logits, sleeve_labels
            ) + media_criterion(media_logits, media_labels)
            if (
                ev_logits is not None
                and evidence_criterion is not None
                and "evidence_label" in batch
            ):
                ev_labels = batch["evidence_label"].to(self.device)
                loss = loss + self.media_evidence_aux_weight * evidence_criterion(
                    ev_logits, ev_labels
                )

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(
        self,
        loader: DataLoader,
    ) -> dict[str, dict]:
        """
        Run one evaluation epoch without gradient computation.

        Returns:
            Dict mapping target → {y_true, y_pred, y_proba, item_ids}
        """
        self.model.eval()

        sleeve_preds, sleeve_probas, sleeve_true = [], [], []
        media_preds, media_probas, media_true = [], [], []
        item_ids = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            _out = self.model(input_ids, attention_mask)
            if len(_out) == 3:
                sleeve_logits, media_logits, _ = _out
            else:
                sleeve_logits, media_logits = _out

            sleeve_proba = torch.softmax(sleeve_logits, dim=-1).cpu().numpy()
            media_proba = torch.softmax(media_logits, dim=-1).cpu().numpy()

            sleeve_probas.append(sleeve_proba)
            media_probas.append(media_proba)

            sleeve_preds.append(sleeve_proba.argmax(axis=1))
            media_preds.append(media_proba.argmax(axis=1))

            sleeve_true.append(batch["sleeve_label"].numpy())
            media_true.append(batch["media_label"].numpy())
            item_ids.extend(batch["item_id"])

        return {
            "sleeve": {
                "y_true": np.concatenate(sleeve_true),
                "y_pred": np.concatenate(sleeve_preds),
                "y_proba": np.concatenate(sleeve_probas),
                "item_ids": item_ids,
            },
            "media": {
                "y_true": np.concatenate(media_true),
                "y_pred": np.concatenate(media_preds),
                "y_proba": np.concatenate(media_probas),
            },
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        sleeve_criterion: nn.CrossEntropyLoss,
        media_criterion: nn.CrossEntropyLoss,
        evidence_criterion: Optional[nn.CrossEntropyLoss] = None,
    ) -> dict:
        """
        Full training loop with early stopping on val macro-F1.

        Early stopping monitors the mean of sleeve and media val
        macro-F1 — both targets must improve together.

        Returns:
            Dict with best val metrics and epoch of best checkpoint.
        """
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        optimizer = AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_f1 = -1.0
        best_epoch = 0
        patience_count = 0
        best_state_dict = None
        history = []

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(
                train_loader,
                optimizer,
                scheduler,
                sleeve_criterion,
                media_criterion,
                evidence_criterion=evidence_criterion,
            )

            val_results = self.eval_epoch(val_loader)

            val_sleeve_f1 = f1_score(
                val_results["sleeve"]["y_true"],
                val_results["sleeve"]["y_pred"],
                average="macro",
                zero_division=0,
            )
            val_media_f1 = f1_score(
                val_results["media"]["y_true"],
                val_results["media"]["y_pred"],
                average="macro",
                zero_division=0,
            )
            val_mean_f1 = (val_sleeve_f1 + val_media_f1) / 2.0

            logger.info(
                "Epoch %2d/%d | loss: %.4f | "
                "val sleeve F1: %.4f | val media F1: %.4f | mean: %.4f",
                epoch,
                self.epochs,
                train_loss,
                val_sleeve_f1,
                val_media_f1,
                val_mean_f1,
            )

            if not self._skip_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_sleeve_f1": val_sleeve_f1,
                        "val_media_f1": val_media_f1,
                        "val_mean_f1": val_mean_f1,
                    },
                    step=epoch,
                )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_sleeve_f1": val_sleeve_f1,
                    "val_media_f1": val_media_f1,
                    "val_mean_f1": val_mean_f1,
                }
            )

            if val_mean_f1 > best_val_f1:
                best_val_f1 = val_mean_f1
                best_epoch = epoch
                patience_count = 0
                best_state_dict = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                logger.info("New best — saving checkpoint at epoch %d.", epoch)
            else:
                patience_count += 1
                logger.info(
                    "No improvement (%d/%d patience).",
                    patience_count,
                    self.patience,
                )
                if patience_count >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d. "
                        "Best epoch: %d (val mean F1: %.4f).",
                        epoch,
                        best_epoch,
                        best_val_f1,
                    )
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info("Restored best checkpoint from epoch %d.", best_epoch)

        return {
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "history": history,
        }


class TransformerTrainer(
    _TransformerData,
    _TransformerTraining,
    _TransformerPredictIo,
    _TransformerRun,
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
