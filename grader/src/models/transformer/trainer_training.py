"""Training loop: epochs, early stopping, validation metrics."""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class TransformerTrainingMixin:
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
