"""Inference path: preprocess → model → (caller applies rule engine)."""

from __future__ import annotations

import logging
from typing import Optional

from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer

logger = logging.getLogger(__name__)


class PipelineInferenceMixin:
    """Single/batch prediction and baseline/transformer dispatch."""

    config_path: str
    artifacts_dir: Path
    infer_model: str
    _baseline: Optional[BaselineModel]
    _transformer: Optional[TransformerTrainer]
    _tfidf: Optional[TFIDFFeatureBuilder]

    def predict(
        self,
        text: str,
        item_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Run inference on a single raw text input.

        Handles all preprocessing internally. Callers pass raw text
        (voice-transcribed or typed) and receive a structured prediction.

        Args:
            text:     raw seller notes or user description
            item_id:  optional item identifier for output
            metadata: optional dict with additional context
                      (source, media_verifiable, etc.)

        Returns:
            Final prediction dict with rule engine applied.
        """
        results = self.predict_batch(
            texts=[text],
            item_ids=[item_id] if item_id else None,
            metadata_list=[metadata] if metadata else None,
        )
        return results[0]

    def predict_batch(
        self,
        texts: list[str],
        item_ids: Optional[list[str]] = None,
        metadata_list: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Run inference on a batch of raw text inputs.

        Steps:
          1. Preprocess each text (normalize, expand abbreviations)
          2. Run model prediction (baseline or transformer)
          3. Apply rule engine post-processing
          4. Return final predictions

        Args:
            texts:         list of raw text strings
            item_ids:      optional list of item IDs
            metadata_list: optional list of metadata dicts per item

        Returns:
            List of final prediction dicts with rule engine applied.
        """
        if not texts:
            return []

        if item_ids is None:
            item_ids = [str(i) for i in range(len(texts))]

        if metadata_list is None:
            metadata_list = [{} for _ in texts]

        preprocessor = self._get_preprocessor()
        rule_engine = self._get_rule_engine()

        clean_texts = []
        records = []

        for i, (text, meta) in enumerate(zip(texts, metadata_list)):
            media_verifiable = preprocessor.detect_unverified_media(text)
            media_evidence_strength = preprocessor.detect_media_evidence_strength(
                text
            )
            text_clean = preprocessor.clean_text(text)
            clean_texts.append(text_clean)

            record = {
                "item_id": item_ids[i],
                "text": text,
                "text_clean": text_clean,
                "media_verifiable": media_verifiable,
                "media_evidence_strength": media_evidence_strength,
                "source": meta.get("source", "user_input"),
                **meta,
            }
            record.update(
                preprocessor.compute_description_quality(
                    text,
                    text_clean,
                    sleeve_label=str(meta.get("sleeve_label") or ""),
                    media_label=str(meta.get("media_label") or ""),
                )
            )
            records.append(record)

        predictions = self._model_predict(
            clean_texts=clean_texts,
            item_ids=item_ids,
            records=records,
        )
        self._merge_description_metadata(predictions, records)

        final_predictions = rule_engine.apply_batch(
            predictions=predictions,
            texts=clean_texts,
        )

        return final_predictions

    @staticmethod
    def _merge_description_metadata(
        predictions: list[dict],
        records: list[dict],
    ) -> None:
        """Copy note-adequacy fields from preprocess records into model metadata."""
        keys = (
            "sleeve_note_adequate",
            "media_note_adequate",
            "adequate_for_training",
            "needs_richer_note",
            "description_quality_gaps",
            "description_quality_prompts",
        )
        for pred, rec in zip(predictions, records):
            meta = pred.setdefault("metadata", {})
            for k in keys:
                if k in rec:
                    meta[k] = rec[k]

    def _model_predict(
        self,
        clean_texts: list[str],
        item_ids: list[str],
        records: list[dict],
    ) -> list[dict]:
        """
        Dispatch prediction to the configured model (baseline or transformer).
        Loads model artifacts lazily on first call.
        """
        if self.infer_model == "transformer":
            return self._transformer_predict(clean_texts, item_ids, records)
        return self._baseline_predict(clean_texts, item_ids, records)

    def _transformer_predict(
        self,
        clean_texts: list[str],
        item_ids: list[str],
        records: list[dict],
    ) -> list[dict]:
        """Load transformer artifacts and run prediction."""
        if self._transformer is None:
            logger.info("Loading transformer model from artifacts ...")
            self._transformer = TransformerTrainer(
                config_path=self.config_path
            )
            self._transformer.encoders = (
                self._transformer.load_encoders()
            )
            self._transformer.load_model()

        return self._transformer.predict(
            texts=clean_texts,
            item_ids=item_ids,
            records=records,
        )

    def _baseline_predict(
        self,
        clean_texts: list[str],
        item_ids: list[str],
        records: list[dict],
    ) -> list[dict]:
        """
        Load baseline artifacts and run prediction.
        Vectorizes text using the fitted TF-IDF vectorizer.
        """
        if self._baseline is None:
            logger.info("Loading baseline model from artifacts ...")
            self._baseline = BaselineModel(config_path=self.config_path)
            self._baseline.encoders = self._baseline.load_encoders()

            for target in ["sleeve", "media"]:
                cal_path = (
                    self.artifacts_dir
                    / f"baseline_{target}_calibrated.pkl"
                )
                self._baseline.calibrated[target] = (
                    BaselineModel.load_model(str(cal_path))
                )

        tfidf = self._get_tfidf()

        sleeve_vectorizer = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_sleeve.pkl")
        )
        media_vectorizer = TFIDFFeatureBuilder.load_vectorizer(
            str(self.artifacts_dir / "tfidf_vectorizer_media.pkl")
        )
        X_sleeve = tfidf.transform_records(
            vectorizer=sleeve_vectorizer,
            records=records,
            target="sleeve",
            split="inference",
        )
        X_media = tfidf.transform_records(
            vectorizer=media_vectorizer,
            records=records,
            target="media",
            split="inference",
        )

        return self._baseline.predict(
            X_sleeve=X_sleeve,
            X_media=X_media,
            item_ids=item_ids,
            records=records,
        )
