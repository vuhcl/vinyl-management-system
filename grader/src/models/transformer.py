"""
grader/src/models/transformer.py

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

import contextlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from grader.src.evaluation.metrics import compute_metrics, log_metrics_to_mlflow
from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    is_remote_mlflow_tracking_uri,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TARGETS = ["sleeve", "media"]
SPLITS = ["train", "val", "test"]

EVIDENCE_STRENGTH_TO_IDX: dict[str, int] = {
    "none": 0,
    "weak": 1,
    "strong": 2,
}
_META_HPARAM_KEYS = frozenset({"description", "name"})


def merge_transformer_hparams(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Merge transformer YAML subsection; nested media_evidence_aux combined."""
    out = dict(base)
    for k, v in overrides.items():
        if k in _META_HPARAM_KEYS:
            continue
        if k == "media_evidence_aux" and isinstance(v, dict):
            inner = dict(out.get("media_evidence_aux") or {})
            inner.update(v)
            out[k] = inner
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def resolve_model_artifact_dir(
    artifacts_dir: Path,
    artifact_subdir: Optional[str],
) -> Path:
    """
    Directory for transformer weights relative to ``paths.artifacts``.

    ``artifact_subdir`` should be a *relative* tail such as ``tuning/foo``.
    If someone passes the full ``grader/artifacts`` path again, it would
    double-resolve to ``.../grader/artifacts/grader/artifacts`` — strip
    the redundant prefix when it matches ``artifacts_dir``'s path parts.
    """
    ad = artifacts_dir
    if not artifact_subdir or not str(artifact_subdir).strip():
        return ad
    s = Path(artifact_subdir.strip())
    if s.is_absolute():
        return s.resolve()
    ar_parts = ad.parts
    sub_parts = s.parts
    if len(sub_parts) >= len(ar_parts) and sub_parts[: len(ar_parts)] == ar_parts:
        tail = sub_parts[len(ar_parts) :]
        return ad if not tail else ad / Path(*tail)
    return ad / s


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Select compute device with MPS priority for Apple Silicon.
    Falls back to CUDA then CPU.
    """
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon) for training.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA for training.")
        return torch.device("cuda")
    else:
        logger.info("Using CPU for training.")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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

        # Pre-encode all labels — avoids repeated encoder calls in __getitem__
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# TransformerTrainer
# ---------------------------------------------------------------------------
class TransformerTrainer:
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
    ) -> None:
        self.config = self._load_yaml(config_path)
        if transformer_overrides:
            self.config["models"]["transformer"] = merge_transformer_hparams(
                self.config["models"]["transformer"],
                transformer_overrides,
            )

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
        self.config_path_out = self.model_artifact_dir / "transformer_config.json"
        self.tokenizer_dir = self.model_artifact_dir / "tokenizer"

        self.device = get_device()
        self._skip_mlflow: bool = False

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        configure_mlflow_from_config(self.config)

        self.model: Optional[TwoHeadClassifier] = None
        self.tokenizer: Optional[DistilBertTokenizerFast] = None
        self.encoders: dict = {}

    # -----------------------------------------------------------------------
    # Config and data loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

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

    # -----------------------------------------------------------------------
    # Class weights for imbalanced training
    # -----------------------------------------------------------------------
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

        # Avoid division by zero for classes with no training samples
        counts = np.maximum(counts, 1.0)
        weights = len(labels) / (n_classes * counts)
        weights = weights / weights.sum() * n_classes  # normalize

        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    # -----------------------------------------------------------------------
    # Training epoch
    # -----------------------------------------------------------------------
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

            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # -----------------------------------------------------------------------
    # Evaluation epoch
    # -----------------------------------------------------------------------
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

            # Probabilities via softmax
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

    # -----------------------------------------------------------------------
    # Training loop with early stopping
    # -----------------------------------------------------------------------
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
        # Only optimize parameters that require gradients
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

            # Compute val macro-F1 for both targets
            val_sleeve_f1 = float(
                np.mean(
                    val_results["sleeve"]["y_true"]
                    == val_results["sleeve"]["y_pred"]
                )
            )
            # Use proper macro-F1 via metrics module
            from sklearn.metrics import f1_score

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

            # Early stopping check
            if val_mean_f1 > best_val_f1:
                best_val_f1 = val_mean_f1
                best_epoch = epoch
                patience_count = 0
                # Deep copy state dict — saves best checkpoint in memory
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

        # Restore best checkpoint
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info("Restored best checkpoint from epoch %d.", best_epoch)

        return {
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "history": history,
        }

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
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

        # Process in batches
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
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

    # -----------------------------------------------------------------------
    # Artifact persistence
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(
        self,
        eval_results: dict,
        training_summary: dict,
    ) -> None:
        mlflow.log_params(
            {
                "model_type": "distilbert_two_head",
                "base_model": self.base_model,
                "freeze_encoder": self.freeze_encoder,
                "unfreeze_top_n_layers": self.unfreeze_top_n_layers,
                "media_evidence_aux": self.media_evidence_aux_enabled,
                "media_evidence_aux_weight": self.media_evidence_aux_weight,
                "max_length": self.max_length,
                "dropout": self.dropout,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "epochs_run": training_summary["best_epoch"],
                "early_stopping": self.patience,
                "class_weight": self.class_weight,
                "weight_decay": self.weight_decay,
                "warmup_ratio": self.warmup_ratio,
            }
        )

        mlflow.log_metric("best_val_mean_f1", training_summary["best_val_f1"])

        # Per-split per-target metrics
        for split, target_results in eval_results.items():
            for target, metrics in target_results.items():
                log_metrics_to_mlflow(metrics, prefix="transformer")

        # Model artifacts
        track_uri = mlflow.get_tracking_uri() or ""
        remote = is_remote_mlflow_tracking_uri(track_uri)

        if remote:
            # Upload the servable bundle first so flaky HTTP runs do not leave a
            # run with loose files but no vinyl_grader/ MLmodel (registry/FastAPI).
            try:
                from grader.src.models.grader_pyfunc import log_pyfunc_model

                log_pyfunc_model(self)
                logger.info(
                    "Remote MLflow URI — logged vinyl_grader pyfunc only "
                    "(weights/config/tokenizer/encoders are inside that bundle)."
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pyfunc logging failed (%s); falling back to loose artifacts "
                    "(no standalone weights on remote — too large to duplicate).",
                    exc,
                )
            mlflow.log_artifact(str(self.config_path_out))
            mlflow.log_artifacts(str(self.tokenizer_dir), artifact_path="tokenizer")
            for target in ("sleeve", "media"):
                enc_path = self.artifacts_dir / f"label_encoder_{target}.pkl"
                if enc_path.exists():
                    mlflow.log_artifact(str(enc_path))
            return

        mlflow.log_artifact(str(self.weights_path))
        mlflow.log_artifact(str(self.config_path_out))
        mlflow.log_artifacts(str(self.tokenizer_dir), artifact_path="tokenizer")
        for target in ("sleeve", "media"):
            enc_path = self.artifacts_dir / f"label_encoder_{target}.pkl"
            if enc_path.exists():
                mlflow.log_artifact(str(enc_path))

        try:
            from grader.src.models.grader_pyfunc import log_pyfunc_model

            log_pyfunc_model(self)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pyfunc model logging skipped: %s", exc)

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(
        self,
        dry_run: bool = False,
        skip_mlflow: bool = False,
        mlflow_run_name: Optional[str] = None,
    ) -> dict:
        """
        Full transformer training pipeline.

        Args:
            dry_run: one epoch, no save.
            skip_mlflow: no MLflow run or metric logging.
            mlflow_run_name: optional MLflow run name.

        Returns:
            model, encoders, eval, training (best epoch / val F1).
        """
        self._skip_mlflow = bool(skip_mlflow or dry_run)
        run_name = mlflow_run_name or "transformer_distilbert_two_head"
        mlflow_ctx = (
            contextlib.nullcontext()
            if self._skip_mlflow
            else mlflow.start_run(run_name=run_name)
        )

        with mlflow_ctx:

            # Load data
            split_records = {
                split: self._load_jsonl(self.split_paths[split])
                for split in SPLITS
            }
            self.encoders = self.load_encoders()

            # Initialize tokenizer and model
            logger.info("Loading tokenizer and model: %s", self.base_model)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                self.base_model
            )
            unfreeze_n = (
                self.unfreeze_top_n_layers if self.freeze_encoder else 0
            )
            n_evidence = 3 if self.media_evidence_aux_enabled else 0
            self.model = TwoHeadClassifier(
                n_sleeve_classes=len(self.encoders["sleeve"].classes_),
                n_media_classes=len(self.encoders["media"].classes_),
                dropout=self.dropout,
                freeze_encoder=self.freeze_encoder,
                unfreeze_top_n_layers=unfreeze_n,
                n_evidence_classes=n_evidence,
                base_model=self.base_model,
            ).to(self.device)

            # Build datasets and loaders
            datasets = {
                split: VinylGraderDataset(
                    records=split_records[split],
                    tokenizer=self.tokenizer,
                    sleeve_encoder=self.encoders["sleeve"],
                    media_encoder=self.encoders["media"],
                    max_length=self.max_length,
                    include_evidence=self.media_evidence_aux_enabled,
                )
                for split in SPLITS
            }

            loaders = {
                "train": DataLoader(
                    datasets["train"],
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,  # 0 = main process — safe on MPS
                ),
                "val": DataLoader(
                    datasets["val"],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                ),
                "test": DataLoader(
                    datasets["test"],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                ),
            }

            # Class weights
            sleeve_weights = self._compute_class_weights(
                split_records["train"], "sleeve", self.encoders["sleeve"]
            )
            media_weights = self._compute_class_weights(
                split_records["train"], "media", self.encoders["media"]
            )

            sleeve_criterion = nn.CrossEntropyLoss(weight=sleeve_weights)
            media_criterion = nn.CrossEntropyLoss(weight=media_weights)
            evidence_criterion = (
                nn.CrossEntropyLoss()
                if self.media_evidence_aux_enabled
                else None
            )

            # Override epochs for dry run
            if dry_run:
                self.epochs = 1
                self.patience = 1

            # Train
            training_summary = self.train(
                loaders["train"],
                loaders["val"],
                sleeve_criterion,
                media_criterion,
                evidence_criterion=evidence_criterion,
            )

            # Evaluate on all splits
            eval_results: dict = {}
            for split in SPLITS:
                logger.info("--- Evaluating on %s split ---", split.upper())
                raw = self.eval_epoch(loaders[split])
                eval_results[split] = {}

                for target in TARGETS:
                    metrics = compute_metrics(
                        y_true=raw[target]["y_true"],
                        y_pred=raw[target]["y_pred"],
                        y_proba=raw[target]["y_proba"],
                        class_names=self.encoders[target].classes_,
                        target=target,
                        split=split,
                    )
                    # Keep probabilities for pipeline calibration (not in compute_metrics return)
                    eval_results[split][target] = {
                        **metrics,
                        "y_proba": raw[target]["y_proba"],
                    }
                    logger.info(
                        "Evaluation — split=%s target=%s | "
                        "macro-F1: %.4f | accuracy: %.4f | ECE: %.4f",
                        split,
                        target,
                        metrics["macro_f1"],
                        metrics["accuracy"],
                        metrics["ece"],
                    )

            test_thin_path = (
                Path(self.config["paths"]["splits"]) / "test_thin.jsonl"
            )
            if test_thin_path.exists():
                thin_records = self._load_jsonl(test_thin_path)
                if thin_records:
                    logger.info("--- Evaluating on TEST_THIN split ---")
                    thin_ds = VinylGraderDataset(
                        records=thin_records,
                        tokenizer=self.tokenizer,
                        sleeve_encoder=self.encoders["sleeve"],
                        media_encoder=self.encoders["media"],
                        max_length=self.max_length,
                        include_evidence=self.media_evidence_aux_enabled,
                    )
                    thin_loader = DataLoader(
                        thin_ds,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=0,
                    )
                    raw_thin = self.eval_epoch(thin_loader)
                    eval_results["test_thin"] = {}
                    for target in TARGETS:
                        metrics = compute_metrics(
                            y_true=raw_thin[target]["y_true"],
                            y_pred=raw_thin[target]["y_pred"],
                            y_proba=raw_thin[target]["y_proba"],
                            class_names=self.encoders[target].classes_,
                            target=target,
                            split="test_thin",
                        )
                        eval_results["test_thin"][target] = {
                            **metrics,
                            "y_proba": raw_thin[target]["y_proba"],
                        }
                        logger.info(
                            "Evaluation — split=test_thin target=%s | "
                            "macro-F1: %.4f | accuracy: %.4f | ECE: %.4f",
                            target,
                            metrics["macro_f1"],
                            metrics["accuracy"],
                            metrics["ece"],
                        )

            # Summary
            for target in TARGETS:
                msg = (
                    "RESULTS SUMMARY — target=%s | "
                    "train macro-F1: %.4f | "
                    "val macro-F1:   %.4f | "
                    "test macro-F1:  %.4f"
                ) % (
                    target,
                    eval_results["train"][target]["macro_f1"],
                    eval_results["val"][target]["macro_f1"],
                    eval_results["test"][target]["macro_f1"],
                )
                if "test_thin" in eval_results:
                    msg += " | test_thin macro-F1: %.4f" % (
                        eval_results["test_thin"][target]["macro_f1"],
                    )
                logger.info(msg)

            if dry_run:
                logger.info(
                    "Dry run — skipping artifact saves and MLflow logging."
                )
                return {
                    "model": self.model,
                    "encoders": self.encoders,
                    "eval": eval_results,
                    "training": training_summary,
                }

            self.save_model()
            run_id = ""
            if not skip_mlflow:
                run_id = mlflow.active_run().info.run_id if mlflow.active_run() else ""
                self._log_mlflow(eval_results, training_summary)

        return {
            "model": self.model,
            "encoders": self.encoders,
            "eval": eval_results,
            "training": training_summary,
            "mlflow_run_id": run_id,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train two-head DistilBERT classifier "
        "for vinyl condition grading"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train for 1 epoch without saving artifacts",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Unfreeze encoder for full fine-tuning (ablation)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Train and save artifacts without MLflow",
    )
    args = parser.parse_args()

    trainer = TransformerTrainer(
        config_path=args.config,
        freeze_encoder_override=not args.unfreeze if args.unfreeze else None,
    )
    trainer.run(dry_run=args.dry_run, skip_mlflow=args.skip_mlflow)


if __name__ == "__main__":
    main()
