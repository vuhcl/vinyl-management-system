"""DistilBERT encoder with sleeve/media (and optional evidence) heads."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import DistilBertModel

logger = logging.getLogger(__name__)


class TwoHeadClassifier(nn.Module):
    """
    DistilBERT encoder with two independent classification heads.

    Both heads receive the same [CLS] token embedding from the
    shared encoder. Independent dropout and linear layers produce
    logits for sleeve and media targets respectively.

    Args:
        n_sleeve_classes: number of canonical sleeve grades
        n_media_classes:  number of canonical media grades
        dropout:                 dropout probability before each head
        freeze_encoder:        if False, full fine-tune
        unfreeze_top_n_layers: if freeze_encoder, unfreeze last N blocks (0=all frozen)
        n_evidence_classes:    >0 enables aux head for media evidence strength
        base_model:            HuggingFace model name or local path
    """

    def __init__(
        self,
        n_sleeve_classes: int,
        n_media_classes: int,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        unfreeze_top_n_layers: int = 0,
        n_evidence_classes: int = 0,
        base_model: str = "distilbert-base-uncased",
    ) -> None:
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained(base_model)
        self.n_evidence_classes = n_evidence_classes

        if not freeze_encoder:
            logger.info("Encoder unfrozen — full fine-tuning enabled.")
        elif unfreeze_top_n_layers <= 0:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(
                "Encoder frozen — only classification heads will be trained."
            )
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
            layers = self.encoder.transformer.layer
            n = min(int(unfreeze_top_n_layers), len(layers))
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(
                "Encoder partially unfrozen — last %d transformer layer(s).",
                n,
            )

        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT

        self.sleeve_dropout = nn.Dropout(dropout)
        self.sleeve_head = nn.Linear(hidden_size, n_sleeve_classes)

        self.media_dropout = nn.Dropout(dropout)
        self.media_head = nn.Linear(hidden_size, n_media_classes)

        self.evidence_dropout: Optional[nn.Dropout] = None
        self.evidence_head: Optional[nn.Linear] = None
        if n_evidence_classes > 0:
            self.evidence_dropout = nn.Dropout(dropout)
            self.evidence_head = nn.Linear(hidden_size, n_evidence_classes)
            nn.init.normal_(self.evidence_head.weight, std=0.02)
            nn.init.zeros_(self.evidence_head.bias)

        nn.init.normal_(self.sleeve_head.weight, std=0.02)
        nn.init.zeros_(self.sleeve_head.bias)
        nn.init.normal_(self.media_head.weight, std=0.02)
        nn.init.zeros_(self.media_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]

        sleeve_logits = self.sleeve_head(self.sleeve_dropout(cls_output))
        media_logits = self.media_head(self.media_dropout(cls_output))

        if self.evidence_head is not None and self.evidence_dropout is not None:
            ev_logits = self.evidence_head(self.evidence_dropout(cls_output))
            return sleeve_logits, media_logits, ev_logits
        return sleeve_logits, media_logits
