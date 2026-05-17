"""Pickle save/load and cold-start loading from artifacts dir."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Tuple

from .constants import SPLITS, TARGETS

logger = logging.getLogger(__name__)


class BaselineArtifactsMixin:
    """Persist heads/calibrators and reload for evaluation-only workflows."""

    artifacts_dir: Path
    model_paths: dict
    calibrated_paths: dict
    models: dict
    calibrated: dict
    config: dict

    def save_models(self) -> None:
        """Save both raw and calibrated models to artifacts/."""
        for target in TARGETS:
            with open(self.model_paths[target], "wb") as f:
                pickle.dump(self.models[target], f)
            with open(self.calibrated_paths[target], "wb") as f:
                pickle.dump(self.calibrated[target], f)
            logger.info("Saved models — target=%s", target)

    @staticmethod
    def load_model(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_trained_from_artifacts(cls, config_path: str) -> Tuple[Any, dict]:
        """
        Load pickled baseline heads + encoders from ``paths.artifacts`` and
        evaluate train/val/test using on-disk TF-IDF features.

        Used when skipping training (e.g. Colab after local baseline train).
        Expects the same artifacts as ``run()`` would write.
        """
        inst = cls(config_path=config_path)
        inst.encoders = inst.load_encoders()
        for target in TARGETS:
            raw_path = inst.model_paths[target]
            cal_path = inst.calibrated_paths[target]
            if not raw_path.is_file() or not cal_path.is_file():
                raise FileNotFoundError(
                    f"Missing baseline artifact(s) for {target}: "
                    f"{raw_path} and/or {cal_path} — train baseline locally first "
                    "or copy artifacts into paths.artifacts."
                )
            inst.models[target] = cls.load_model(str(raw_path))
            inst.calibrated[target] = cls.load_model(str(cal_path))

        features = inst.load_all_features()
        split_records: dict[str, list[dict]] = {}
        splits_dir = Path(inst.config["paths"]["splits"])
        for split in SPLITS:
            path = splits_dir / f"{split}.jsonl"
            records: list[dict] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            split_records[split] = records
        if "test_thin" in features:
            tt_path = splits_dir / "test_thin.jsonl"
            thin_recs: list[dict] = []
            if tt_path.exists():
                with open(tt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            thin_recs.append(json.loads(line))
            split_records["test_thin"] = thin_recs

        eval_results: dict[str, dict[str, dict]] = {}
        for split in SPLITS:
            eval_results[split] = inst.evaluate(
                features,
                split,
                records=split_records.get(split),
            )
        if "test_thin" in features:
            eval_results["test_thin"] = inst.evaluate(
                features,
                "test_thin",
                records=split_records.get("test_thin"),
            )

        bundle = {
            "models": inst.models,
            "calibrated": inst.calibrated,
            "eval": eval_results,
        }
        thin_note = (
            ", ".join((*SPLITS, "test_thin"))
            if "test_thin" in eval_results
            else ", ".join(SPLITS)
        )
        logger.info(
            "Loaded baseline from artifacts — evaluated %s (no training).",
            thin_note,
        )
        return inst, bundle
