"""TF-IDF feature loading and grade ordinal map from guidelines."""

from __future__ import annotations

import logging
from pathlib import Path

from grader.src.config_io import load_yaml_mapping
from grader.src.features.tfidf_features import TFIDFFeatureBuilder

from .constants import SPLITS, TARGETS

logger = logging.getLogger(__name__)


class BaselineFeaturesMixin:
    """Load matrices, encoders, and rubric ordering for baseline heads."""

    config: dict
    features_dir: Path
    artifacts_dir: Path

    def _load_grade_ordinal_map(self) -> dict[str, int]:
        rules_cfg = self.config.get("rules", {})
        guidelines_path = rules_cfg.get("guidelines_path")
        if not guidelines_path:
            return {}
        p = Path(guidelines_path)
        if not p.exists():
            return {}
        g = load_yaml_mapping(p)
        return {
            str(k): int(v)
            for k, v in g.get("grade_ordinal_map", {}).items()
            if isinstance(v, int)
        }

    def load_all_features(
        self,
    ) -> dict[str, dict[str, dict]]:
        """
        Load TF-IDF feature matrices and label arrays for all
        splits and targets from artifacts/features/.

        Returns nested dict: features[split][target] = {"X": ..., "y": ...}
        """
        features: dict = {split: {} for split in SPLITS}

        for split in SPLITS:
            for target in TARGETS:
                X, y = TFIDFFeatureBuilder.load_features(
                    str(self.features_dir), split, target
                )
                features[split][target] = {"X": X, "y": y}
                logger.info(
                    "Loaded features — split=%s target=%s shape=%s",
                    split,
                    target,
                    X.shape,
                )

        thin_x = self.features_dir / "test_thin_sleeve_X.npz"
        if thin_x.exists():
            features["test_thin"] = {}
            for target in TARGETS:
                X, y = TFIDFFeatureBuilder.load_features(
                    str(self.features_dir), "test_thin", target
                )
                features["test_thin"][target] = {"X": X, "y": y}
                logger.info(
                    "Loaded features — split=test_thin target=%s shape=%s",
                    target,
                    X.shape,
                )

        return features

    def load_encoders(self) -> dict:
        """Load fitted label encoders from artifacts/."""
        encoders = {}
        for target in TARGETS:
            path = self.artifacts_dir / f"label_encoder_{target}.pkl"
            encoders[target] = TFIDFFeatureBuilder.load_encoder(str(path))
            logger.info(
                "Loaded label encoder — target=%s classes=%s",
                target,
                list(encoders[target].classes_),
            )
        return encoders
