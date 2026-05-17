"""Helpers for rule-engine evaluation and offline split loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer
from grader.src.rules.rule_engine import RuleEngine
from grader.src.schemas import GraderPrediction, merge_description_quality_metadata


class PipelineRuleEvalMixin:
    """Baseline-from-features, rule-eval dispatch, JSONL split loading."""

    config: dict
    artifacts_dir: Path

    def _baseline_predict_from_features(
        self,
        baseline: BaselineModel,
        split: str = "test",
    ) -> list[GraderPrediction]:
        """
        Rebuild baseline predictions from saved TF-IDF features for a split.
        Used when the transformer is not run (rule eval uses the same path).
        """
        X_sleeve, _ = TFIDFFeatureBuilder.load_features(
            str(self.artifacts_dir / "features"),
            split=split,
            target="sleeve",
        )
        X_media, _ = TFIDFFeatureBuilder.load_features(
            str(self.artifacts_dir / "features"),
            split=split,
            target="media",
        )
        split_records = self._load_split(split)
        item_ids = [
            r.get("item_id", str(i)) for i, r in enumerate(split_records)
        ]

        return baseline.predict(
            X_sleeve=X_sleeve,
            X_media=X_media,
            item_ids=item_ids,
            records=split_records,
        )

    def _predictions_for_rule_eval(
        self,
        split: str,
        trainer: Optional[TransformerTrainer],
        baseline: BaselineModel,
        use_transformer: bool,
    ) -> tuple[list[GraderPrediction], list[str]]:
        """Model-only predictions and aligned text for rule evaluation."""
        records = self._load_split(split)
        texts = [r.get("text_clean") or r.get("text", "") for r in records]
        item_ids = [r.get("item_id") for r in records]
        if use_transformer:
            if trainer is None:
                raise RuntimeError("use_transformer=True but trainer is None")
            preds = trainer.predict(
                texts=texts,
                item_ids=item_ids,
                records=records,
            )
        else:
            preds = self._baseline_predict_from_features(baseline, split=split)
        merge_description_quality_metadata(preds, records)
        return preds, texts

    def _run_rule_engine_evaluation(
        self,
        rule_engine: RuleEngine,
        trainer: Optional[TransformerTrainer],
        baseline: BaselineModel,
        use_transformer: bool,
    ) -> tuple[dict, dict[str, float], dict[str, str]]:
        """
        Rule-adjusted metrics, model-only metrics, and override audit on
        test and test_thin (when split + features exist).

        Also writes ``grade_analysis_{split}.txt`` (including a
        ``RULE-OWNED SLICE`` banner + stratified override-audit section)
        and a dual-format baseline snapshot to
        ``rule_engine_baseline.json`` plus MLflow tags keyed
        ``rule_baseline_*`` — see §8 of the rule-owned eval plan.
        """
        from grader.src.evaluation.rule_engine_eval import (
            run_rule_engine_evaluation,
        )

        return run_rule_engine_evaluation(
            self,
            rule_engine,
            trainer,
            baseline,
            use_transformer,
        )

    def _load_split(self, split: str) -> list[dict]:
        """Load records from a split JSONL file."""
        splits_dir = Path(self.config["paths"]["splits"])
        path = splits_dir / f"{split}.jsonl"
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
