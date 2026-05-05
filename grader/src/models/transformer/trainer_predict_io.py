"""Inference batching and artifact save/load."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast

from .constants import EVIDENCE_STRENGTH_TO_IDX
from .two_head import TwoHeadClassifier

logger = logging.getLogger(__name__)


class TransformerPredictIoMixin:
    """predict(), save_model(), load_model()."""

    model: Optional[nn.Module]
    tokenizer: Optional[DistilBertTokenizerFast]
    encoders: dict
    device: torch.device
    batch_size: int
    max_length: int
    base_model: str
    freeze_encoder: bool
    unfreeze_top_n_layers: int
    dropout: float
    weights_path: Path
    config_path_out: Path
    tokenizer_dir: Path
    model_artifact_dir: Path

    def predict(
        self,
        texts: list[str],
        item_ids: Optional[list[str]] = None,
        records: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Run inference on a list of raw text strings.
        Returns structured prediction dicts matching the output schema.

        Rule engine is NOT applied — that is pipeline.py's responsibility.

        Args:
            texts:    list of seller note strings
            item_ids: optional list of item IDs for output
            records:  optional source records for metadata fields

        Returns:
            List of prediction dicts with confidence scores.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Run train() or load_model() first."
            )

        if item_ids is None:
            item_ids = [str(i) for i in range(len(texts))]

        self.model.eval()
        all_sleeve_preds = []
        all_sleeve_probas = []
        all_media_preds = []
        all_media_probas = []
        track_evidence = getattr(self.model, "evidence_head", None) is not None
        all_ev_probas: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                _out = self.model(input_ids, attention_mask)
                if len(_out) == 3:
                    sleeve_logits, media_logits, ev_logits = _out
                else:
                    sleeve_logits, media_logits = _out
                    ev_logits = None

                sleeve_proba = (
                    torch.softmax(sleeve_logits, dim=-1).cpu().numpy()
                )
                media_proba = torch.softmax(media_logits, dim=-1).cpu().numpy()

                all_sleeve_probas.append(sleeve_proba)
                all_media_probas.append(media_proba)
                all_sleeve_preds.append(sleeve_proba.argmax(axis=1))
                all_media_preds.append(media_proba.argmax(axis=1))
                if track_evidence and ev_logits is not None:
                    all_ev_probas.append(
                        torch.softmax(ev_logits, dim=-1).cpu().numpy()
                    )

        sleeve_probas = np.concatenate(all_sleeve_probas)
        media_probas = np.concatenate(all_media_probas)
        sleeve_preds = np.concatenate(all_sleeve_preds)
        media_preds = np.concatenate(all_media_preds)
        ev_probas_concat = (
            np.concatenate(all_ev_probas) if all_ev_probas else None
        )

        sleeve_classes = self.encoders["sleeve"].classes_
        media_classes = self.encoders["media"].classes_
        ev_names = [
            k
            for k, _ in sorted(
                EVIDENCE_STRENGTH_TO_IDX.items(), key=lambda x: x[1]
            )
        ]

        predictions = []
        for i in range(len(texts)):
            sleeve_label = self.encoders["sleeve"].inverse_transform(
                [sleeve_preds[i]]
            )[0]
            media_label = self.encoders["media"].inverse_transform(
                [media_preds[i]]
            )[0]

            sleeve_scores = {
                cls: round(float(sleeve_probas[i][j]), 4)
                for j, cls in enumerate(sleeve_classes)
            }
            media_scores = {
                cls: round(float(media_probas[i][j]), 4)
                for j, cls in enumerate(media_classes)
            }

            record = records[i] if records else {}
            media_verifiable = record.get("media_verifiable", True)
            contradiction = record.get("contradiction_detected", False)
            source = record.get("source", "unknown")

            meta = {
                "source": source,
                "media_verifiable": media_verifiable,
                "rule_override_applied": False,
                "rule_override_target": None,
                "contradiction_detected": contradiction,
            }
            if ev_probas_concat is not None:
                row = ev_probas_concat[i]
                meta["media_evidence_scores"] = {
                    ev_names[j]: round(float(row[j]), 4)
                    for j in range(len(ev_names))
                }

            predictions.append(
                {
                    "item_id": item_ids[i],
                    "predicted_sleeve_condition": sleeve_label,
                    "predicted_media_condition": media_label,
                    "confidence_scores": {
                        "sleeve": sleeve_scores,
                        "media": media_scores,
                    },
                    "metadata": meta,
                }
            )

        return predictions

    def save_model(self) -> None:
        """
        Save model weights, config, and tokenizer to artifacts/.

        Saves:
          - transformer_weights.pt       (state dict)
          - transformer_config.json      (architecture config)
          - tokenizer/                   (HuggingFace tokenizer files)
        """
        torch.save(self.model.state_dict(), self.weights_path)

        n_ev = int(
            getattr(self.model, "n_evidence_classes", 0)
            if self.model is not None
            else 0
        )
        model_config = {
            "base_model": self.base_model,
            "freeze_encoder": self.freeze_encoder,
            "unfreeze_top_n_layers": self.unfreeze_top_n_layers,
            "max_length": self.max_length,
            "dropout": self.dropout,
            "n_sleeve_classes": len(self.encoders["sleeve"].classes_),
            "n_media_classes": len(self.encoders["media"].classes_),
            "n_evidence_classes": n_ev,
        }
        with open(self.config_path_out, "w") as f:
            json.dump(model_config, f, indent=2)

        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(self.tokenizer_dir))

        logger.info("Model artifacts saved to %s", self.model_artifact_dir)

    def load_model(self, config_path_out: Optional[str] = None) -> None:
        """
        Load model weights and tokenizer from artifacts/.
        Requires label encoders to already be loaded.
        """
        cfg_path = Path(config_path_out or self.config_path_out)
        with open(cfg_path, "r") as f:
            model_config = json.load(f)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            str(self.tokenizer_dir)
        )
        n_ev = int(model_config.get("n_evidence_classes", 0))
        unfreeze_n = int(model_config.get("unfreeze_top_n_layers", 0))
        self.model = TwoHeadClassifier(
            n_sleeve_classes=model_config["n_sleeve_classes"],
            n_media_classes=model_config["n_media_classes"],
            dropout=model_config["dropout"],
            freeze_encoder=model_config["freeze_encoder"],
            unfreeze_top_n_layers=unfreeze_n,
            n_evidence_classes=n_ev,
            base_model=model_config["base_model"],
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.eval()
        logger.info("Model loaded from %s", self.model_artifact_dir)
