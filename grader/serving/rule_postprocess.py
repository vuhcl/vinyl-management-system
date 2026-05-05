"""
Apply the same preprocessing + RuleEngine path as grader.src.pipeline.Pipeline
after MLflow pyfunc (model-only) inference.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from grader.src.config_io import load_yaml
from grader.src.data.preprocess import Preprocessor
from grader.src.rules.rule_engine import RuleEngine
from grader.src.schemas import GraderPrediction, merge_description_quality_metadata

logger = logging.getLogger(__name__)

_SERVING_DIR = Path(__file__).resolve().parent
_GRADER_ROOT = _SERVING_DIR.parent

_preprocessor: Preprocessor | None = None
_rule_engine: RuleEngine | None = None


def _default_config_path() -> Path:
    return _GRADER_ROOT / "configs" / "grader.yaml"


def _default_guidelines_path() -> Path:
    return _GRADER_ROOT / "configs" / "grading_guidelines.yaml"


def _resolved_config_paths() -> tuple[str, str]:
    cfg = os.environ.get("GRADER_CONFIG_PATH", "").strip()
    gl = os.environ.get("GRADER_GUIDELINES_PATH", "").strip()
    if not cfg:
        cfg = str(_default_config_path())
    if not gl:
        gl = str(_default_guidelines_path())
    return cfg, gl


def init_rule_stack() -> None:
    """
    Load Preprocessor + RuleEngine (fail-fast if YAML is missing).

    Optional env:
        GRADER_CONFIG_PATH — default ``<grader>/configs/grader.yaml``
        GRADER_GUIDELINES_PATH — default
        ``<grader>/configs/grading_guidelines.yaml``
    """
    global _preprocessor, _rule_engine
    cfg, gl = _resolved_config_paths()
    if not Path(cfg).is_file():
        raise RuntimeError(
            f"Grader config not found: {cfg} "
            "(set GRADER_CONFIG_PATH or mount grader/configs)"
        )
    if not Path(gl).is_file():
        raise RuntimeError(
            f"Grading guidelines not found: {gl} "
            "(set GRADER_GUIDELINES_PATH or mount grader/configs)"
        )
    _preprocessor = Preprocessor(config_path=cfg, guidelines_path=gl)
    _raw = load_yaml(cfg)
    _cfg = _raw if isinstance(_raw, dict) else {}
    _rules = _cfg.get("rules") or {}
    _allow_ex = bool(_rules.get("allow_excellent_soft_override", False))
    _rule_engine = RuleEngine(
        guidelines_path=gl,
        allow_excellent_soft_override=_allow_ex,
    )
    logger.info("RuleEngine + Preprocessor loaded for API post-processing")


def get_preprocessor() -> Preprocessor:
    if _preprocessor is None:
        raise RuntimeError(
            "Rule stack not initialized; call init_rule_stack()"
        )
    return _preprocessor


def get_rule_engine() -> RuleEngine:
    if _rule_engine is None:
        raise RuntimeError(
            "Rule stack not initialized; call init_rule_stack()"
        )
    return _rule_engine


def preprocess_batch(
    texts: list[str],
    item_ids: list[str],
    metadata_list: list[dict[str, Any]],
) -> tuple[list[str], list[dict]]:
    """Mirror Pipeline.predict_batch step 1 (raw → clean + records)."""
    pre = get_preprocessor()
    clean_texts: list[str] = []
    records: list[dict] = []
    for i, (text, meta) in enumerate(zip(texts, metadata_list)):
        media_verifiable = pre.detect_unverified_media(text)
        media_evidence_strength = pre.detect_media_evidence_strength(text)
        text_clean = pre.clean_text(text)
        clean_texts.append(text_clean)
        record: dict = {
            "item_id": item_ids[i],
            "text": text,
            "text_clean": text_clean,
            "media_verifiable": media_verifiable,
            "media_evidence_strength": media_evidence_strength,
            "source": meta.get("source", "grader_api"),
            **meta,
        }
        record.update(
            pre.compute_description_quality(
                text,
                text_clean,
                sleeve_label=str(meta.get("sleeve_label") or ""),
                media_label=str(meta.get("media_label") or ""),
            )
        )
        records.append(record)
    return clean_texts, records


def _pyfunc_df_to_prediction_dicts(
    out_df: pd.DataFrame,
    records: list[dict],
) -> list[GraderPrediction]:
    """
    Build rule-engine prediction dicts from pyfunc output.

    Pyfunc only exposes top-1 confidence per target; RuleEngine uses the
    score for the predicted grade only (``scores.get(predicted_grade)``).
    """
    predictions: list[GraderPrediction] = []
    for (_, row), rec in zip(out_df.iterrows(), records):
        ps = str(row["predicted_sleeve_condition"])
        pm = str(row["predicted_media_condition"])
        sc = float(row["sleeve_confidence"])
        mc = float(row["media_confidence"])
        predictions.append(
            {
                "item_id": row["item_id"],
                "predicted_sleeve_condition": ps,
                "predicted_media_condition": pm,
                "confidence_scores": {
                    "sleeve": {ps: sc},
                    "media": {pm: mc},
                },
                "metadata": {
                    "source": rec.get("source", "grader_api"),
                    "media_verifiable": rec.get("media_verifiable", True),
                    "rule_override_applied": False,
                    "rule_override_target": None,
                    "contradiction_detected": False,
                },
            }
        )
    return predictions


def apply_rules_to_pyfunc_batch(
    out_df: pd.DataFrame,
    raw_texts: list[str],
    item_ids: list[Any],
    metadata_list: list[dict[str, Any]],
) -> list[GraderPrediction]:
    """
    Run preprocessing + RuleEngine on pyfunc rows.

    Args:
        out_df: DataFrame from ``pyfunc.predict``
            (text + item_id + predictions).
        raw_texts: Same texts passed into the model (strip only; preprocessor
            receives these as in ``Pipeline.predict_batch``).
        item_ids: Aligned item ids (any JSON-serializable type).
        metadata_list: Per-row extra metadata (may be empty dicts).
    """
    ids = [str(x) for x in item_ids]
    clean_texts, records = preprocess_batch(raw_texts, ids, metadata_list)
    predictions = _pyfunc_df_to_prediction_dicts(out_df, records)
    merge_description_quality_metadata(predictions, records)
    return get_rule_engine().apply_batch(predictions, clean_texts)
