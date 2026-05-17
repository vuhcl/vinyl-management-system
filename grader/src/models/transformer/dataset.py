"""PyTorch Dataset for two-head DistilBERT training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

from .constants import EVIDENCE_STRENGTH_TO_IDX


class VinylGraderDataset(Dataset):
    """
    PyTorch Dataset for vinyl grader text classification.

    Tokenizes text_clean on the fly using DistilBERT tokenizer.
    Returns tensors for input_ids, attention_mask, and both labels.

    Args:
        records:    list of preprocessed record dicts
        tokenizer:  fitted DistilBertTokenizerFast
        sleeve_encoder: fitted LabelEncoder for sleeve target
        media_encoder:  fitted LabelEncoder for media target
        max_length: maximum token sequence length
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer: DistilBertTokenizerFast,
        sleeve_encoder,
        media_encoder,
        max_length: int = 128,
        include_evidence: bool = False,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.sleeve_encoder = sleeve_encoder
        self.media_encoder = media_encoder
        self.max_length = max_length
        self.include_evidence = include_evidence

        sleeve_labels = [r["sleeve_label"] for r in records]
        media_labels = [r["media_label"] for r in records]
        self.sleeve_y = torch.tensor(
            sleeve_encoder.transform(sleeve_labels), dtype=torch.long
        )
        self.media_y = torch.tensor(
            media_encoder.transform(media_labels), dtype=torch.long
        )
        if include_evidence:
            ev = [
                EVIDENCE_STRENGTH_TO_IDX.get(
                    str(r.get("media_evidence_strength", "none")).lower(),
                    0,
                )
                for r in records
            ]
            self.evidence_y = torch.tensor(ev, dtype=torch.long)
        else:
            self.evidence_y = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        text = self.records[idx].get("text_clean") or self.records[idx].get(
            "text", ""
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        out = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sleeve_label": self.sleeve_y[idx],
            "media_label": self.media_y[idx],
            "item_id": self.records[idx].get("item_id", str(idx)),
        }
        if self.include_evidence and self.evidence_y is not None:
            out["evidence_label"] = self.evidence_y[idx]
        return out
