"""Logistic-regression head training, optional tuning, and auxiliary heads."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .constants import TARGETS

logger = logging.getLogger(__name__)


class BaselineTrainMixin:
    """Build and fit baseline heads, boundary blend, and media aux."""

    C: float
    max_iter: int
    tol: float
    class_weight: str
    solver: str
    random_state: int
    tuning_enabled: bool
    tuning_c_values: list[float]
    boundary_enabled: bool
    boundary_alpha: float
    media_evidence_aux_enabled: bool
    encoders: dict
    selected_c: dict[str, float]
    boundary_models: dict[str, LogisticRegression]
    boundary_band_labels: dict[str, np.ndarray]
    media_evidence_aux_model: Optional[LogisticRegression]

    def build_model(
        self,
        target: str,
        *,
        C: Optional[float] = None,
    ) -> LogisticRegression:
        """
        Construct a LogisticRegression head for a single target.
        Hyperparameters come from grader.yaml — nothing hardcoded.
        """
        return LogisticRegression(
            C=self.C if C is None else C,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )

    @staticmethod
    def _macro_f1(y_true, y_pred) -> float:
        return float(f1_score(y_true, y_pred, average="macro"))

    def _tune_c_for_target(self, target: str, features: dict) -> float:
        """
        Grid-search C on train->val for a single target.
        Uses macro-F1 on val as selection metric.
        """
        X_train = features["train"][target]["X"]
        y_train = features["train"][target]["y"]
        X_val = features["val"][target]["X"]
        y_val = features["val"][target]["y"]

        best_c = self.C
        best_val_f1 = -1.0

        for c in self.tuning_c_values:
            model = self.build_model(target, C=c)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_f1 = self._macro_f1(y_val, val_pred)
            logger.info(
                "Tune baseline C — target=%s C=%.4f val macro-F1=%.4f",
                target,
                c,
                val_f1,
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_c = c

        logger.info(
            "Selected baseline C — target=%s C=%.4f (best val macro-F1=%.4f)",
            target,
            best_c,
            best_val_f1,
        )
        return best_c

    def train(
        self,
        features: dict,
    ) -> dict[str, LogisticRegression]:
        """
        Fit one LogisticRegression head per target on train features.

        Args:
            features: nested dict from load_all_features()

        Returns:
            Dict mapping target → fitted LogisticRegression.
        """
        models = {}
        for target in TARGETS:
            logger.info("Training baseline head — target=%s ...", target)
            X_train = features["train"][target]["X"]
            y_train = features["train"][target]["y"]

            selected_c = self.C
            if self.tuning_enabled:
                selected_c = self._tune_c_for_target(target, features)
            self.selected_c[target] = selected_c

            model = self.build_model(target, C=selected_c)
            model.fit(X_train, y_train)
            models[target] = model

            train_preds = model.predict(X_train)
            train_f1 = f1_score(y_train, train_preds, average="macro")
            logger.info("Train macro-F1 — target=%s: %.4f", target, train_f1)

        return models

    @staticmethod
    def _grade_to_band(grade: str) -> str:
        if grade in {"Poor", "Good"}:
            return "low"
        if grade in {"Very Good", "Very Good Plus"}:
            return "mid"
        if grade in {"Excellent", "Near Mint", "Mint"}:
            return "high"
        return "other"

    def train_boundary_models(self, features: dict) -> None:
        """Train coarse-bin auxiliary heads for boundary-aware blending."""
        if not self.boundary_enabled:
            return
        self.boundary_models = {}
        self.boundary_band_labels = {}
        for target in TARGETS:
            X_train = features["train"][target]["X"]
            y_train = features["train"][target]["y"]
            encoder = self.encoders[target]
            grades = encoder.inverse_transform(y_train)
            band_labels = np.asarray([self._grade_to_band(g) for g in grades])
            bmodel = LogisticRegression(
                C=self.selected_c.get(target, self.C),
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=self.class_weight,
                solver=self.solver,
                random_state=self.random_state,
            )
            bmodel.fit(X_train, band_labels)
            self.boundary_models[target] = bmodel
            self.boundary_band_labels[target] = bmodel.classes_
            logger.info(
                "Boundary auxiliary head trained — target=%s bands=%s",
                target,
                list(bmodel.classes_),
            )

    def train_media_evidence_aux(
        self,
        features: dict,
        split_records: dict[str, list[dict]],
    ) -> None:
        """Train auxiliary classifier for media evidence strength."""
        if not self.media_evidence_aux_enabled:
            self.media_evidence_aux_model = None
            return
        train_records = split_records.get("train", [])
        X_train = features["train"]["media"]["X"]
        y = np.asarray(
            [
                str(r.get("media_evidence_strength", "none")).lower()
                for r in train_records
            ]
        )
        if len(y) != X_train.shape[0]:
            logger.warning(
                "Skipping media evidence aux head: record/feature mismatch."
            )
            self.media_evidence_aux_model = None
            return
        model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight="balanced",
            solver=self.solver,
            random_state=self.random_state,
        )
        model.fit(X_train, y)
        self.media_evidence_aux_model = model
        logger.info(
            "Media evidence aux head trained — classes=%s",
            list(model.classes_),
        )
