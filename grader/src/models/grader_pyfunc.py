"""
grader/src/models/grader_pyfunc.py

MLflow PythonModel wrapper for the two-head DistilBERT vinyl grader.

Logged with MLflow models-from-code (``python_model`` = path to
``vinyl_grader_pyfunc_entry.py``) to avoid CloudPickle / ``artifact_path``
deprecation warnings. The implementation embeds a minimal inference-only
``TwoHeadClassifier`` so the bundle stays self-contained at serve time.

Artifacts expected in context.artifacts:
    transformer_weights  — transformer_weights.pt
    transformer_config   — transformer_config.json
    tokenizer_dir        — directory with HuggingFace tokenizer files
    encoder_sleeve       — label_encoder_sleeve.pkl
    encoder_media        — label_encoder_media.pkl

Input schema (pd.DataFrame):
    text          str     seller note (required)
    item_id       any     optional, echoed through to output

Output schema (pd.DataFrame):
    item_id                     — echoed if present, else 0-based index
    predicted_sleeve_condition  — grade string (e.g. "Very Good Plus")
    predicted_media_condition   — grade string
    sleeve_confidence           — softmax probability of top class (0-1)
    media_confidence            — softmax probability of top class (0-1)

Optional inference calibration (does not change labels unless logits are nearly tied):
    GRADER_SOFTMAX_TEMPERATURE — divisor applied to logits before softmax on both heads (default 1.0).
    GRADER_SLEEVE_SOFTMAX_TEMPERATURE — overrides sleeve head only (falls back to shared/default).
    GRADER_MEDIA_SOFTMAX_TEMPERATURE — overrides media head only (falls back to shared/default).

Usage:
    # Serve registered model version 3:
    mlflow models serve -m "models:/VinylGrader/3" --port 5001

    # Invoke:
    curl http://127.0.0.1:5001/invocations \\
      -H "Content-Type: application/json" \\
      -d '{"dataframe_records": [{"text": "VG+ sleeve, light marks"}]}'
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_LENGTH = 128


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Softmax(logits / T). T > 1 spreads mass (lower peak confidence); T == 1 is standard softmax.

    Invalid or non-positive T is treated as 1.0 for backward-compatible behavior.
    """
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        t = 1.0
    return torch.softmax(logits / t, dim=-1)


def _temperature_from_env(specific_key: str, shared_key: str) -> float:
    """Prefer head-specific env, then GRADER_SOFTMAX_TEMPERATURE, then 1.0."""
    for key in (specific_key, shared_key):
        raw = os.environ.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if math.isfinite(v) and v > 0.0:
            return v
    return 1.0


def _torch_load_state_dict_cpu(path: str) -> Any:
    """
    Load a checkpoint onto CPU regardless of save device (MPS/CUDA).

    ``map_location=torch.device('cpu')`` is not always enough for legacy
    ``transformer_weights.pt`` files on Linux/Docker (MPS is unknown there).
    """
    return torch.load(
        path,
        map_location=lambda storage, _: storage,
        weights_only=False,
    )


# ---------------------------------------------------------------------------
# Minimal self-contained TwoHeadClassifier (inference only)
# ---------------------------------------------------------------------------


class _TwoHeadClassifier(nn.Module):
    """
    DistilBERT encoder with sleeve + media classification heads.
    Kept self-contained so the pyfunc model has no project-level imports.
    """

    def __init__(
        self,
        n_sleeve_classes: int,
        n_media_classes: int,
        dropout: float = 0.3,
        n_evidence_classes: int = 0,
        base_model: str = "distilbert-base-uncased",
    ) -> None:
        super().__init__()
        from transformers import DistilBertModel

        self.encoder = DistilBertModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size

        self.sleeve_dropout = nn.Dropout(dropout)
        self.sleeve_head = nn.Linear(hidden_size, n_sleeve_classes)

        self.media_dropout = nn.Dropout(dropout)
        self.media_head = nn.Linear(hidden_size, n_media_classes)

        self.evidence_head: Optional[nn.Linear] = None
        self.evidence_dropout: Optional[nn.Dropout] = None
        if n_evidence_classes > 0:
            self.evidence_dropout = nn.Dropout(dropout)
            self.evidence_head = nn.Linear(hidden_size, n_evidence_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0, :]
        s = self.sleeve_head(self.sleeve_dropout(cls))
        m = self.media_head(self.media_dropout(cls))
        if self.evidence_head is not None and self.evidence_dropout is not None:
            return s, m, self.evidence_head(self.evidence_dropout(cls))
        return s, m


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MLflow PythonModel
# ---------------------------------------------------------------------------


class VinylGraderModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for the vinyl grader transformer.

    Loaded via mlflow.pyfunc.load_model() or mlflow models serve.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from transformers import DistilBertTokenizerFast

        arts = context.artifacts

        with open(arts["transformer_config"]) as f:
            cfg = json.load(f)

        self.device = _get_device()
        self.max_length = int(cfg.get("max_length", _DEFAULT_MAX_LENGTH))
        self.batch_size = _DEFAULT_BATCH_SIZE

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            arts["tokenizer_dir"]
        )

        with open(arts["encoder_sleeve"], "rb") as f:
            self.enc_sleeve = pickle.load(f)
        with open(arts["encoder_media"], "rb") as f:
            self.enc_media = pickle.load(f)

        # Build on CPU, load weights with map_location=cpu so checkpoints saved on
        # MPS/CUDA unpickle on Linux/Docker (no MPS; CUDA optional).
        self.model = _TwoHeadClassifier(
            n_sleeve_classes=cfg["n_sleeve_classes"],
            n_media_classes=cfg["n_media_classes"],
            dropout=float(cfg.get("dropout", 0.3)),
            n_evidence_classes=int(cfg.get("n_evidence_classes", 0)),
            base_model=cfg["base_model"],
        )
        state = _torch_load_state_dict_cpu(arts["transformer_weights"])
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._temp_sleeve = _temperature_from_env(
            "GRADER_SLEEVE_SOFTMAX_TEMPERATURE",
            "GRADER_SOFTMAX_TEMPERATURE",
        )
        self._temp_media = _temperature_from_env(
            "GRADER_MEDIA_SOFTMAX_TEMPERATURE",
            "GRADER_SOFTMAX_TEMPERATURE",
        )
        logger.info(
            "VinylGraderModel loaded — device=%s "
            "sleeve_classes=%d media_classes=%d "
            "softmax_temperature sleeve=%.4f media=%.4f",
            self.device,
            cfg["n_sleeve_classes"],
            cfg["n_media_classes"],
            self._temp_sleeve,
            self._temp_media,
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        texts = model_input["text"].tolist()
        item_ids = (
            model_input["item_id"].tolist()
            if "item_id" in model_input.columns
            else list(range(len(texts)))
        )

        all_sleeve_preds: list[int] = []
        all_sleeve_conf: list[float] = []
        all_media_preds: list[int] = []
        all_media_conf: list[float] = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                end = i + self.batch_size
                batch = texts[i:end]
                enc = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                out = self.model(input_ids, attention_mask)
                s_proba = softmax_with_temperature(
                    out[0], self._temp_sleeve
                ).cpu().numpy()
                m_proba = softmax_with_temperature(
                    out[1], self._temp_media
                ).cpu().numpy()

                all_sleeve_preds.extend(s_proba.argmax(axis=1).tolist())
                all_sleeve_conf.extend(s_proba.max(axis=1).tolist())
                all_media_preds.extend(m_proba.argmax(axis=1).tolist())
                all_media_conf.extend(m_proba.max(axis=1).tolist())

        sleeve_grades = self.enc_sleeve.inverse_transform(all_sleeve_preds)
        media_grades = self.enc_media.inverse_transform(all_media_preds)
        sleeve_conf = [round(float(c), 4) for c in all_sleeve_conf]
        media_conf = [round(float(c), 4) for c in all_media_conf]
        return pd.DataFrame(
            {
                "item_id": item_ids,
                "predicted_sleeve_condition": sleeve_grades,
                "predicted_media_condition": media_grades,
                "sleeve_confidence": sleeve_conf,
                "media_confidence": media_conf,
            }
        )


_SERVING_REQUIREMENTS = [
    "torch>=2.2.0",
    "transformers>=4.38.0",
    "numpy>=1.26.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.8.0",
]


def log_pyfunc_model(trainer: Any) -> str:
    """
    Log the current trainer state as an MLflow pyfunc model.

    Must be called inside an active ``mlflow.start_run()`` context.
    Uses ``name=`` and a path-based ``python_model`` (models-from-code)
    so MLflow does not emit CloudPickle / ``artifact_path`` deprecation
    warnings.

    Args:
        trainer: a ``TransformerTrainer`` instance whose model has been trained
                 and whose artifacts are saved to disk.

    Returns:
        The artifact URI of the logged model (e.g. ``runs:/<id>/vinyl_grader``).
    """
    artifacts = {
        "transformer_weights": str(trainer.weights_path),
        "transformer_config": str(trainer.config_path_out),
        "tokenizer_dir": str(trainer.tokenizer_dir),
        "encoder_sleeve": str(
            trainer.artifacts_dir / "label_encoder_sleeve.pkl"
        ),
        "encoder_media": str(
            trainer.artifacts_dir / "label_encoder_media.pkl"
        ),
    }

    input_example = pd.DataFrame(
        {"text": ["Mint sleeve, still in shrink. Vinyl unplayed."]}
    )

    _models_dir = Path(__file__).resolve().parent
    _entry = _models_dir / "vinyl_grader_pyfunc_entry.py"
    _impl = _models_dir / "grader_pyfunc.py"

    mlflow.pyfunc.log_model(
        name="vinyl_grader",
        python_model=os.fspath(_entry),
        code_paths=[os.fspath(_impl)],
        artifacts=artifacts,
        input_example=input_example,
        pip_requirements=_SERVING_REQUIREMENTS,
    )

    run_id = mlflow.active_run().info.run_id
    uri = f"runs:/{run_id}/vinyl_grader"
    logger.info("Logged pyfunc model → %s", uri)
    return uri
