"""MLflow logging, full training orchestration, and predict I/O."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from grader.src.evaluation.metrics import compute_metrics, log_metrics_to_mlflow
from grader.src.mlflow_tracking import (
    is_remote_mlflow_tracking_uri,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
)
from grader.src.schemas import GraderPrediction

from .constants import EVIDENCE_STRENGTH_TO_IDX, SPLITS, TARGETS
from .dataset import VinylGraderDataset
from .two_head import TwoHeadClassifier

logger = logging.getLogger(__name__)


class _TransformerPredictIo:
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
    ) -> list[GraderPrediction]:
        """
        Run inference on a list of raw text strings.
        Returns structured prediction dicts matching the output schema.

        Rule engine is NOT applied — that is the inference pipeline's job.

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


class _TransformerRun:
    """_log_mlflow and run()."""

    config: dict
    artifacts_dir: Path
    model_artifact_dir: Path
    weights_path: Path
    config_path_out: Path
    tokenizer_dir: Path
    split_paths: dict
    base_model: str
    freeze_encoder: bool
    unfreeze_top_n_layers: int
    dropout: float
    max_length: int
    batch_size: int
    epochs: int
    patience: int
    lr: float
    media_evidence_aux_enabled: bool
    media_evidence_aux_weight: float
    class_weight: str
    weight_decay: float
    warmup_ratio: float
    device: torch.device
    model: Optional[nn.Module]
    tokenizer: Optional[DistilBertTokenizerFast]
    encoders: dict
    _skip_mlflow: bool

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

        for split, target_results in eval_results.items():
            for target, metrics in target_results.items():
                log_metrics_to_mlflow(metrics, prefix="transformer")

        if not mlflow_log_artifacts_enabled(self.config):
            mlflow.log_param("mlflow_log_artifacts", "false")
            logger.info(
                "MLflow metrics-only (mlflow.log_artifacts: false) — "
                "skipping artifact and pyfunc uploads."
            )
            return

        track_uri = mlflow.get_tracking_uri() or ""
        remote = is_remote_mlflow_tracking_uri(track_uri)

        if remote:
            try:
                from grader.src.models.grader_pyfunc import log_pyfunc_model

                log_pyfunc_model(self)
                logger.info(
                    "Remote MLflow URI — logged vinyl_grader pyfunc only "
                    "(weights/config/tokenizer/encoders are inside that "
                    "bundle)."
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pyfunc logging failed (%s); falling back to loose "
                    "artifacts (no standalone weights on remote — too large "
                    "to duplicate).",
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
            logger.warning("pyfunc logging failed: %s", exc)

    def run(
        self,
        dry_run: bool = False,
        skip_mlflow: bool = False,
        mlflow_run_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Full transformer training pipeline.

        Args:
            dry_run: one epoch, no save.
            skip_mlflow: no MLflow run or metric logging.
            mlflow_run_name: optional MLflow run name.

        Returns:
            model, encoders, eval, training (best epoch / val F1).
        """
        self._skip_mlflow = bool(
            skip_mlflow or dry_run or not mlflow_enabled(self.config)
        )
        run_name = mlflow_run_name or "transformer_distilbert_two_head"
        mlflow_ctx = (
            contextlib.nullcontext()
            if self._skip_mlflow
            else mlflow.start_run(run_name=run_name)
        )

        with mlflow_ctx:

            split_records = {
                split: self._load_jsonl(self.split_paths[split])
                for split in SPLITS
            }
            self.encoders = self.load_encoders()

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
                    num_workers=0,
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

            if dry_run:
                self.epochs = 1
                self.patience = 1

            training_summary = self.train(
                loaders["train"],
                loaders["val"],
                sleeve_criterion,
                media_criterion,
                evidence_criterion=evidence_criterion,
            )

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
                    "mlflow_run_id": (
                        mlflow.active_run().info.run_id
                        if mlflow.active_run()
                        else ""
                    ),
                }

            self.save_model()
            run_id = ""
            if not self._skip_mlflow:
                run_id = (
                    mlflow.active_run().info.run_id
                    if mlflow.active_run()
                    else ""
                )
                self._log_mlflow(eval_results, training_summary)

        return {
            "model": self.model,
            "encoders": self.encoders,
            "eval": eval_results,
            "training": training_summary,
            "mlflow_run_id": run_id,
        }
