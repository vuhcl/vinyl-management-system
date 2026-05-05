"""JSONL loading and shared label encoders for TransformerTrainer."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from .constants import TARGETS

logger = logging.getLogger(__name__)


class TransformerDataMixin:
    """Split file IO and encoder pickle loading."""

    artifacts_dir: Path
    split_paths: dict

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
