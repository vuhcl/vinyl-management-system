"""
grader/src/models/transformer/

Two-head DistilBERT classifier for vinyl condition grading.
Shared frozen encoder with independent sleeve and media output heads.

Architecture:
    DistilBERT encoder (frozen by default)
        └── [CLS] token embedding (768-dim)
            ├── Dropout → Linear → sleeve logits (n_sleeve_classes)
            └── Dropout → Linear → media logits  (n_media_classes)

Device priority: MPS (Apple Silicon) → CUDA → CPU
Encoder policy: freeze_encoder, unfreeze_top_n_layers (partial blocks),
media_evidence_aux (optional 3-way head). Full FT: freeze_encoder: false.

Label encoders are shared with tfidf_features.py to ensure class
indices are identical across baseline and transformer for comparison.

Output artifacts:
    grader/artifacts/transformer_weights.pt
    grader/artifacts/transformer_config.json
    grader/artifacts/tokenizer/           (HuggingFace tokenizer files)

Usage:
    python -m grader.src.models.transformer
    python -m grader.src.models.transformer --dry-run
    python -m grader.src.models.transformer --unfreeze  (full fine-tuning)
"""

from .dataset import VinylGraderDataset
from .device import get_device
from .hparams import merge_transformer_hparams, resolve_model_artifact_dir
from .trainer_core import TransformerTrainer
from .two_head import TwoHeadClassifier

__all__ = [
    "TransformerTrainer",
    "TwoHeadClassifier",
    "VinylGraderDataset",
    "get_device",
    "merge_transformer_hparams",
    "resolve_model_artifact_dir",
]
