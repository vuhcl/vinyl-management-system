"""
Optional embedding features for advanced models.

- Sentence / transformer embeddings (Phase 2)
- Used by transformer fine-tuning or as fixed embeddings
"""
from typing import Any

import numpy as np


def get_embedding_model(model_name: str = "distilbert-base-uncased") -> Any:
    """
    Load a Hugging Face transformer for embeddings.
    Optional dependency: transformers, torch.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def embed_texts(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute [CLS] or mean-pooled embeddings for a list of texts.
    Returns (n_samples, hidden_size).
    """
    if tokenizer is None or model is None:
        raise ImportError("transformers and torch required for embed_texts")
    import torch
    model.eval()
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
            # mean pool over tokens (ignore padding)
            mask = enc["attention_mask"]
            hidden = out.last_hidden_state
            masked = hidden * mask.unsqueeze(-1)
            pooled = masked.sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-9)
        all_embeds.append(pooled.numpy())
    return np.vstack(all_embeds)
