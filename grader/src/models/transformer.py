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
Encoder freezing controlled by configs/grader.yaml freeze_encoder flag.
Full fine-tuning is an ablation — set freeze_encoder: false.

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
from typing import Optional

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
    configure_mlflow_for_transformer_init,
    mlflow_enabled,
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
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.sleeve_encoder = sleeve_encoder
        self.media_encoder = media_encoder
        self.max_length = max_length

        # Pre-encode all labels — avoids repeated encoder calls in __getitem__
        sleeve_labels = [r["sleeve_label"] for r in records]
        media_labels = [r["media_label"] for r in records]
        self.sleeve_y = torch.tensor(
            sleeve_encoder.transform(sleeve_labels), dtype=torch.long
        )
        self.media_y = torch.tensor(
            media_encoder.transform(media_labels), dtype=torch.long
        )

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

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sleeve_label": self.sleeve_y[idx],
            "media_label": self.media_y[idx],
            "item_id": self.records[idx].get("item_id", str(idx)),
        }


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
        dropout:          dropout probability before each head
        freeze_encoder:   if True, DistilBERT weights are frozen
        base_model:       HuggingFace model name or local path
    """

    def __init__(
        self,
        n_sleeve_classes: int,
        n_media_classes: int,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        base_model: str = "distilbert-base-uncased",
    ) -> None:
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained(base_model)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(
                "Encoder frozen — only classification heads will be trained."
            )
        else:
            logger.info("Encoder unfrozen — full fine-tuning enabled.")

        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT

        self.sleeve_dropout = nn.Dropout(dropout)
        self.sleeve_head = nn.Linear(hidden_size, n_sleeve_classes)

        self.media_dropout = nn.Dropout(dropout)
        self.media_head = nn.Linear(hidden_size, n_media_classes)

        # Initialize head weights with small values for stable training
        nn.init.normal_(self.sleeve_head.weight, std=0.02)
        nn.init.zeros_(self.sleeve_head.bias)
        nn.init.normal_(self.media_head.weight, std=0.02)
        nn.init.zeros_(self.media_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids:      (batch, seq_len) token ID tensor
            attention_mask: (batch, seq_len) attention mask tensor

        Returns:
            Tuple of (sleeve_logits, media_logits)
            each of shape (batch, n_classes).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token is the first token of last_hidden_state
        cls_output = outputs.last_hidden_state[:, 0, :]

        sleeve_logits = self.sleeve_head(self.sleeve_dropout(cls_output))
        media_logits = self.media_head(self.media_dropout(cls_output))

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
        mlflow.tracking_uri
        mlflow.experiment_name
    """

    def __init__(
        self,
        config_path: str,
        freeze_encoder_override: Optional[bool] = None,
        *,
        tuning: bool = False,
    ) -> None:
        self.config = self._load_yaml(config_path)
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

        # Paths
        splits_dir = Path(self.config["paths"]["splits"])
        self.artifacts_dir = Path(self.config["paths"]["artifacts"])
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.split_paths = {
            split: splits_dir / f"{split}.jsonl" for split in SPLITS
        }

        self.weights_path = self.artifacts_dir / "transformer_weights.pt"
        self.config_path_out = self.artifacts_dir / "transformer_config.json"
        self.tokenizer_dir = self.artifacts_dir / "tokenizer"

        # Device
        self.device = get_device()

        # Set seeds for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # MLflow (remote production vs local tuning — see transformer_tune)
        configure_mlflow_for_transformer_init(
            self.config, tuning=self._mlflow_tuning
        )

        # Populated during run()
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
                sleeve_logits, media_logits = self.model(
                    input_ids, attention_mask
                )
            except RuntimeError as e:
                # MPS op not supported — fall back to CPU for this batch
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
                    sleeve_logits, media_logits = self.model(
                        input_ids, attention_mask
                    )
                else:
                    raise

            loss = sleeve_criterion(
                sleeve_logits, sleeve_labels
            ) + media_criterion(media_logits, media_labels)

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

            sleeve_logits, media_logits = self.model(input_ids, attention_mask)

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
        optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=0.01)

        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * 0.1)
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

            if mlflow.active_run() is not None:
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

                sleeve_logits, media_logits = self.model(
                    input_ids, attention_mask
                )

                sleeve_proba = (
                    torch.softmax(sleeve_logits, dim=-1).cpu().numpy()
                )
                media_proba = torch.softmax(media_logits, dim=-1).cpu().numpy()

                all_sleeve_probas.append(sleeve_proba)
                all_media_probas.append(media_proba)
                all_sleeve_preds.append(sleeve_proba.argmax(axis=1))
                all_media_preds.append(media_proba.argmax(axis=1))

        sleeve_probas = np.concatenate(all_sleeve_probas)
        media_probas = np.concatenate(all_media_probas)
        sleeve_preds = np.concatenate(all_sleeve_preds)
        media_preds = np.concatenate(all_media_preds)

        sleeve_classes = self.encoders["sleeve"].classes_
        media_classes = self.encoders["media"].classes_

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

            predictions.append(
                {
                    "item_id": item_ids[i],
                    "predicted_sleeve_condition": sleeve_label,
                    "predicted_media_condition": media_label,
                    "confidence_scores": {
                        "sleeve": sleeve_scores,
                        "media": media_scores,
                    },
                    "metadata": {
                        "source": source,
                        "media_verifiable": media_verifiable,
                        "rule_override_applied": False,
                        "rule_override_target": None,
                        "contradiction_detected": contradiction,
                    },
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

        model_config = {
            "base_model": self.base_model,
            "freeze_encoder": self.freeze_encoder,
            "max_length": self.max_length,
            "dropout": self.dropout,
            "n_sleeve_classes": len(self.encoders["sleeve"].classes_),
            "n_media_classes": len(self.encoders["media"].classes_),
        }
        with open(self.config_path_out, "w") as f:
            json.dump(model_config, f, indent=2)

        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(self.tokenizer_dir))

        logger.info("Model artifacts saved to %s", self.artifacts_dir)

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
        self.model = TwoHeadClassifier(
            n_sleeve_classes=model_config["n_sleeve_classes"],
            n_media_classes=model_config["n_media_classes"],
            dropout=model_config["dropout"],
            freeze_encoder=model_config["freeze_encoder"],
            base_model=model_config["base_model"],
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.eval()
        logger.info("Model loaded from %s", self.artifacts_dir)

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
                "max_length": self.max_length,
                "dropout": self.dropout,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "epochs_run": training_summary["best_epoch"],
                "early_stopping": self.patience,
                "class_weight": self.class_weight,
            }
        )

        mlflow.log_metric("best_val_mean_f1", training_summary["best_val_f1"])

        # Per-split per-target metrics
        for split, target_results in eval_results.items():
            for target, metrics in target_results.items():
                log_metrics_to_mlflow(metrics, prefix="transformer")

        # Model artifacts
        mlflow.log_artifact(str(self.weights_path))
        mlflow.log_artifact(str(self.config_path_out))

        try:
            from grader.src.models.grader_pyfunc import log_pyfunc_model

            log_pyfunc_model(self)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pyfunc logging failed: %s", exc)

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(
        self,
        dry_run: bool = False,
        *,
        mlflow_run_name: str | None = None,
        skip_mlflow: bool = False,
    ) -> dict:
        """
        Full transformer training pipeline:
          1. Load split records and label encoders
          2. Initialize tokenizer and model
          3. Build DataLoaders
          4. Compute class weights
          5. Train with early stopping on val macro-F1
          6. Evaluate on train, val, test splits
          7. Save model artifacts
          8. Log metrics and artifacts to MLflow

        Args:
            dry_run: train for 1 epoch without saving or logging.
            mlflow_run_name: optional run name (e.g. ``tune_preset``).
            skip_mlflow: if True, train without an MLflow run.

        Returns:
            Dict with model, encoders, and eval results.
        """
        run_name = mlflow_run_name or "transformer_distilbert_two_head"
        use_mlflow = mlflow_enabled(self.config) and not skip_mlflow
        ctx = (
            mlflow.start_run(run_name=run_name)
            if use_mlflow
            else contextlib.nullcontext()
        )
        with ctx:

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
            self.model = TwoHeadClassifier(
                n_sleeve_classes=len(self.encoders["sleeve"].classes_),
                n_media_classes=len(self.encoders["media"].classes_),
                dropout=self.dropout,
                freeze_encoder=self.freeze_encoder,
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
                    eval_results[split][target] = metrics
                    logger.info(
                        "Evaluation — split=%s target=%s | "
                        "macro-F1: %.4f | accuracy: %.4f | ECE: %.4f",
                        split,
                        target,
                        metrics["macro_f1"],
                        metrics["accuracy"],
                        metrics["ece"],
                    )

            # Summary
            for target in TARGETS:
                logger.info(
                    "RESULTS SUMMARY — target=%s | "
                    "train macro-F1: %.4f | "
                    "val macro-F1:   %.4f | "
                    "test macro-F1:  %.4f",
                    target,
                    eval_results["train"][target]["macro_f1"],
                    eval_results["val"][target]["macro_f1"],
                    eval_results["test"][target]["macro_f1"],
                )

            if dry_run:
                logger.info(
                    "Dry run — skipping artifact saves and MLflow logging."
                )
                run = mlflow.active_run()
                return {
                    "model": self.model,
                    "encoders": self.encoders,
                    "eval": eval_results,
                    "mlflow_run_id": run.info.run_id if run else "",
                }

            self.save_model()
            if mlflow.active_run() is not None:
                self._log_mlflow(eval_results, training_summary)

            run = mlflow.active_run()
            run_id = run.info.run_id if run else ""
            return {
                "model": self.model,
                "encoders": self.encoders,
                "eval": eval_results,
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
    args = parser.parse_args()

    trainer = TransformerTrainer(
        config_path=args.config,
        freeze_encoder_override=not args.unfreeze if args.unfreeze else None,
    )
    trainer.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
