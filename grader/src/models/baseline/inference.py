"""Calibration, blending, and batch prediction for baseline heads."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .constants import TARGETS
from .isotonic import _IsotonicCalibrator

logger = logging.getLogger(__name__)


class BaselineInferenceMixin:
    """Isotonic calibration and structured prediction dicts."""

    calibrated: dict
    encoders: dict
    boundary_models: dict
    boundary_enabled: bool
    boundary_alpha: float
    media_gating_enabled: bool
    media_gating_alpha_none: float
    media_gating_alpha_weak: float
    confidence_band_enabled: bool
    confidence_band_max_gap: float
    grade_ordinal_map: dict[str, int]
    calibration_method: str
    media_evidence_aux_model: Optional[LogisticRegression]

    def _blend_with_boundary(
        self,
        target: str,
        X,
        base_proba: np.ndarray,
    ) -> np.ndarray:
        if not self.boundary_enabled or target not in self.boundary_models:
            return base_proba
        encoder = self.encoders[target]
        n_classes = len(encoder.classes_)
        band_model = self.boundary_models[target]
        band_proba = band_model.predict_proba(X)
        band_classes = list(band_model.classes_)
        mapped = np.zeros((base_proba.shape[0], n_classes), dtype=float)
        for j, grade in enumerate(encoder.classes_):
            band = self._grade_to_band(str(grade))
            if band in band_classes:
                mapped[:, j] = band_proba[:, band_classes.index(band)]
        blended = (1.0 - self.boundary_alpha) * base_proba + (
            self.boundary_alpha * mapped
        )
        assert base_proba.shape[1] == n_classes, (
            f"base_proba has {base_proba.shape[1]} cols but encoder has {n_classes}"
        )
        row_sum = blended.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return blended / row_sum

    def _predict_with_proba(self, target: str, X) -> tuple[np.ndarray, np.ndarray]:
        proba = self.calibrated[target].predict_proba(X)
        proba = self._blend_with_boundary(target, X, proba)
        pred = proba.argmax(axis=1)
        return pred, proba

    def _apply_media_evidence_gating(
        self,
        proba: np.ndarray,
        media_strengths: Optional[list[str]],
    ) -> np.ndarray:
        if not self.media_gating_enabled or media_strengths is None:
            return proba
        classes = list(self.encoders["media"].classes_)
        conservative = {"Near Mint", "Very Good Plus", "Very Good"}
        prior = np.zeros(len(classes), dtype=np.float64)
        for i, cls in enumerate(classes):
            if cls in conservative:
                prior[i] = 1.0
        if prior.sum() == 0:
            return proba
        prior = prior / prior.sum()
        out = proba.copy()
        n = min(len(media_strengths), out.shape[0])
        for i in range(n):
            s = str(media_strengths[i]).lower()
            if s == "none":
                a = self.media_gating_alpha_none
            elif s == "weak":
                a = self.media_gating_alpha_weak
            else:
                a = 0.0
            if a > 0:
                out[i, :] = (1.0 - a) * out[i, :] + a * prior
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return out / row_sum

    def _confidence_band(self, scores: dict[str, float]) -> Optional[str]:
        if not self.confidence_band_enabled or len(scores) < 2:
            return None
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (g1, p1), (g2, p2) = ranked[0], ranked[1]
        if (p1 - p2) > self.confidence_band_max_gap:
            return None
        if g1 not in self.grade_ordinal_map or g2 not in self.grade_ordinal_map:
            return None
        if abs(self.grade_ordinal_map[g1] - self.grade_ordinal_map[g2]) != 1:
            return None
        ordered = sorted([g1, g2], key=lambda g: self.grade_ordinal_map[g])
        return f"{ordered[0]}/{ordered[1]}"

    def calibrate(
        self,
        models: dict[str, LogisticRegression],
        features: dict,
    ) -> dict[str, _IsotonicCalibrator]:
        """
        Calibrate each fitted model on the val split using per-class isotonic
        regression on the base model's raw predict_proba output.

        This approach preserves the full class set from the base estimator
        (including rare grades like Excellent that may not appear in val),
        avoiding the class-mismatch crash that occurs with CalibratedClassifierCV
        when val lacks some training classes.

        Calibration is fitted on val — NOT train — to prevent overfitting.

        Args:
            models:   fitted LogisticRegression heads
            features: nested dict from load_all_features()

        Returns:
            Dict mapping target → _IsotonicCalibrator wrapper.
        """
        calibrated = {}
        for target in TARGETS:
            logger.info(
                "Calibrating model — target=%s method=%s ...",
                target,
                self.calibration_method,
            )
            X_val = features["val"][target]["X"]
            y_val = features["val"][target]["y"]

            n_classes = len(self.encoders[target].classes_)
            calibrator = _IsotonicCalibrator(models[target], n_classes=n_classes)
            calibrator.fit(X_val, y_val)
            calibrated[target] = calibrator

            logger.info("Calibration complete — target=%s", target)

        return calibrated

    def predict(
        self,
        X_sleeve,
        X_media,
        item_ids: Optional[list[str]] = None,
        records: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Run inference on feature matrices and return structured
        predictions conforming to the agreed output schema.

        Rule engine is NOT applied here — that is pipeline.py's job.
        rule_override_applied is always False at this stage.

        Args:
            X_sleeve:  sparse feature matrix for sleeve target
            X_media:   sparse feature matrix for media target
            item_ids:  optional list of item IDs for output
            records:   optional list of source records for metadata

        Returns:
            List of prediction dicts with confidence scores.
        """
        if not self.calibrated:
            raise RuntimeError(
                "Model not calibrated. Run train() and calibrate() first."
            )

        n_samples = X_sleeve.shape[0]
        if item_ids is None:
            item_ids = [str(i) for i in range(n_samples)]

        sleeve_encoder = self.encoders["sleeve"]
        media_encoder = self.encoders["media"]

        sleeve_pred_idx, sleeve_proba = self._predict_with_proba(
            "sleeve", X_sleeve
        )
        media_pred_idx, media_proba = self._predict_with_proba("media", X_media)

        media_strengths = None
        if records:
            media_strengths = [
                str(r.get("media_evidence_strength", "none")) for r in records
            ]
        media_proba = self._apply_media_evidence_gating(
            media_proba,
            media_strengths,
        )
        media_pred_idx = media_proba.argmax(axis=1)

        sleeve_pred = sleeve_encoder.inverse_transform(sleeve_pred_idx)
        media_pred = media_encoder.inverse_transform(media_pred_idx)

        sleeve_classes = sleeve_encoder.classes_
        media_classes = media_encoder.classes_

        predictions = []
        for i in range(n_samples):
            record = records[i] if records else {}
            media_verifiable = record.get("media_verifiable", True)
            contradiction = record.get("contradiction_detected", False)
            source = record.get("source", "unknown")
            media_strength = record.get("media_evidence_strength", "none")

            sleeve_scores = {
                cls: round(float(sleeve_proba[i][j]), 4)
                for j, cls in enumerate(sleeve_classes)
            }
            media_scores = {
                cls: round(float(media_proba[i][j]), 4)
                for j, cls in enumerate(media_classes)
            }
            sleeve_band = self._confidence_band(sleeve_scores)
            media_band = self._confidence_band(media_scores)
            excellent_proxy_media = 0.0
            nm_key = "Near Mint"
            vgp_key = "Very Good Plus"
            if nm_key in media_scores and vgp_key in media_scores:
                excellent_proxy_media = round(
                    2.0 * min(media_scores[nm_key], media_scores[vgp_key]),
                    4,
                )
            evidence_scores: dict[str, float] = {}
            if self.media_evidence_aux_model is not None:
                eproba = self.media_evidence_aux_model.predict_proba(
                    X_media[i : i + 1]
                )[0]
                for j, cls in enumerate(self.media_evidence_aux_model.classes_):
                    evidence_scores[str(cls)] = round(float(eproba[j]), 4)

            predictions.append(
                {
                    "item_id": item_ids[i],
                    "predicted_sleeve_condition": str(sleeve_pred[i]),
                    "predicted_media_condition": str(media_pred[i]),
                    "confidence_scores": {
                        "sleeve": sleeve_scores,
                        "media": media_scores,
                    },
                    "metadata": {
                        "source": source,
                        "media_verifiable": media_verifiable,
                        "media_evidence_strength": media_strength,
                        "confidence_band_sleeve": sleeve_band,
                        "confidence_band_media": media_band,
                        "excellent_proxy_media": excellent_proxy_media,
                        "media_evidence_scores": evidence_scores,
                        "ambiguous_prediction": bool(
                            sleeve_band is not None or media_band is not None
                        ),
                        "rule_override_applied": False,
                        "rule_override_target": None,
                        "contradiction_detected": contradiction,
                    },
                }
            )

        return predictions
