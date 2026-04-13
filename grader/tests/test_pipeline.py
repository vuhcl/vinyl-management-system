"""
grader/tests/test_pipeline.py
"""

import pytest

from grader.src.pipeline import Pipeline


@pytest.fixture
def pipeline(
    test_config,
    guidelines_path,
    saved_encoder_paths,
    saved_vectorizer_paths,
    saved_feature_paths,
    saved_calibrated_model_paths,
    sample_feature_matrices,
    fitted_encoders,
    fitted_calibrated_models,
):
    """
    Pipeline with baseline inference model pre-loaded.
    fitted_calibrated_models uses raw LR — see conftest.py for rationale.
    """
    pl = Pipeline(
        config_path=test_config,
        guidelines_path=guidelines_path,
    )
    pl.infer_model = "baseline"

    from grader.src.models.baseline import BaselineModel

    bl = BaselineModel(config_path=test_config)
    bl.encoders = fitted_encoders
    bl.models = fitted_calibrated_models
    bl.calibrated = fitted_calibrated_models
    pl._baseline = bl

    return pl


class TestPreprocessingInPipeline:
    def test_raw_text_preprocessed(self, pipeline):
        result = pipeline.predict("NM sleeve, VG+ record, plays fine")
        assert result is not None

    def test_unverified_media_flagged(self, pipeline):
        result = pipeline.predict("untested, sold as seen")
        assert result["metadata"]["media_verifiable"] is False

    def test_verified_media_not_flagged(self, pipeline):
        result = pipeline.predict("plays perfectly, near mint condition")
        assert result["metadata"]["media_verifiable"] is True


class TestPredictMethod:
    def test_returns_dict(self, pipeline):
        result = pipeline.predict("plays perfectly, VG+ sleeve")
        assert isinstance(result, dict)

    def test_output_schema_complete(self, pipeline):
        result = pipeline.predict("plays perfectly, VG+ sleeve")
        for key in [
            "item_id",
            "predicted_sleeve_condition",
            "predicted_media_condition",
            "confidence_scores",
            "metadata",
        ]:
            assert key in result

    def test_confidence_scores_both_targets(self, pipeline):
        result = pipeline.predict("surface noise, light scratches")
        assert "sleeve" in result["confidence_scores"]
        assert "media" in result["confidence_scores"]

    def test_item_id_passed_through(self, pipeline):
        result = pipeline.predict("near mint", item_id="test_999")
        assert result["item_id"] == "test_999"

    def test_predicted_grade_is_string(self, pipeline):
        result = pipeline.predict("plays perfectly")
        assert isinstance(result["predicted_sleeve_condition"], str)
        assert isinstance(result["predicted_media_condition"], str)


class TestPredictBatchMethod:
    def test_returns_list(self, pipeline):
        results = pipeline.predict_batch(
            texts=["plays perfectly", "surface noise", "sealed record"]
        )
        assert isinstance(results, list)

    def test_batch_length_matches_input(self, pipeline):
        texts = ["plays perfectly", "surface noise", "sealed record"]
        results = pipeline.predict_batch(texts=texts)
        assert len(results) == len(texts)

    def test_empty_batch_returns_empty(self, pipeline):
        results = pipeline.predict_batch(texts=[])
        assert results == []

    def test_item_ids_assigned(self, pipeline):
        texts = ["plays fine", "some wear"]
        item_ids = ["id_001", "id_002"]
        results = pipeline.predict_batch(texts=texts, item_ids=item_ids)
        assert results[0]["item_id"] == "id_001"
        assert results[1]["item_id"] == "id_002"


class TestRuleEngineIntegration:
    def test_sealed_does_not_hard_override_mint(self, pipeline):
        """Mint is model-owned; pipeline rule pass does not force Mint from sealed."""
        result = pipeline.predict("factory sealed, never opened")
        assert result["metadata"]["rule_override_applied"] is False

    def test_contradiction_detected(self, pipeline):
        result = pipeline.predict(
            "sealed record with significant surface noise"
        )
        assert result["metadata"]["contradiction_detected"] is True
        assert result["metadata"]["rule_override_applied"] is False

    def test_normal_text_no_override(self, pipeline):
        result = pipeline.predict("light wear on cover, plays fine")
        assert result["metadata"]["contradiction_detected"] is False
