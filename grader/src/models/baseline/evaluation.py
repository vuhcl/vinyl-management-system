"""Metrics, ECE, and confusion-matrix formatting for baseline heads."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .constants import TARGETS

logger = logging.getLogger(__name__)


class BaselineEvaluationMixin:
    """Evaluate splits and render confusion matrices."""

    encoders: dict

    def evaluate(
        self,
        features: dict,
        split: str,
        records: Optional[list[dict]] = None,
    ) -> dict[str, dict]:
        """
        Compute evaluation metrics for both targets on a given split.

        Metrics computed:
          - Macro-F1 (primary)
          - Accuracy
          - Per-class F1, precision, recall
          - Expected Calibration Error (ECE)

        Args:
            features: nested dict from load_all_features()
            split:    "train", "val", "test", or "test_thin" (if features exist)

        Returns:
            Dict mapping target → metrics dict.
        """
        results = {}

        for target in TARGETS:
            X = features[split][target]["X"]
            y_true = features[split][target]["y"]
            encoder = self.encoders[target]

            y_pred, y_proba = self._predict_with_proba(target, X)
            if target == "media":
                strengths = None
                if records is not None:
                    strengths = [
                        str(r.get("media_evidence_strength", "none"))
                        for r in records
                    ]
                y_proba = self._apply_media_evidence_gating(y_proba, strengths)
                y_pred = y_proba.argmax(axis=1)

            macro_f1 = f1_score(y_true, y_pred, average="macro")
            accuracy = accuracy_score(y_true, y_pred)
            ece = self._compute_ece(y_true, y_proba)

            logger.info(
                "Evaluation — split=%s target=%s | "
                "macro-F1: %.4f | accuracy: %.4f | ECE: %.4f",
                split,
                target,
                macro_f1,
                accuracy,
                ece,
            )

            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(len(encoder.classes_))),
                target_names=list(encoder.classes_),
                output_dict=True,
                zero_division=0,
            )

            results[target] = {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "ece": ece,
                "report": report,
            }

        return results

    @staticmethod
    def _compute_ece(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        Measures how well predicted probabilities match actual frequencies.
        Lower is better. 0.0 is perfectly calibrated.

        Uses confidence of predicted class (max probability) for binning.
        """
        confidences = y_proba.max(axis=1)
        correct = (y_proba.argmax(axis=1) == y_true).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if not np.any(mask):
                continue
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def format_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target: str,
    ) -> str:
        """Format a human-readable confusion matrix string."""
        encoder = self.encoders[target]
        classes = encoder.classes_
        all_labels = list(range(len(classes)))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        col_width = max(len(c) for c in classes) + 2
        header = " " * col_width + "".join(
            f"{c:>{col_width}}" for c in classes
        )
        lines = [
            f"CONFUSION MATRIX — {target.upper()}",
            "-" * len(header),
            header,
            "-" * len(header),
        ]
        for i, row_label in enumerate(classes):
            row = f"{row_label:<{col_width}}" + "".join(
                f"{cm[i][j]:>{col_width}}" for j in range(len(classes))
            )
            lines.append(row)
        lines.append("")
        return "\n".join(lines)

    def save_confusion_matrices(self, features: dict) -> None:
        """Save confusion matrices for both targets on test split."""
        for target in TARGETS:
            X = features["test"][target]["X"]
            y_true = features["test"][target]["y"]
            y_pred, _ = self._predict_with_proba(target, X)

            cm_text = self.format_confusion_matrix(y_true, y_pred, target)
            with open(self.confusion_paths[target], "w") as f:
                f.write(cm_text)
            logger.info("Saved confusion matrix — target=%s", target)
