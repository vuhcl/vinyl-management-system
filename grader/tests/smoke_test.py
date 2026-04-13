"""
grader/tests/smoke_test.py

End-to-end smoke tests. Uses synthetic data to verify all modules
wire together correctly without errors.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


class TestSmokeHarmonize:
    def test_harmonize_produces_unified_jsonl(
        self,
        test_config,
        guidelines_path,
        tmp_dirs,
        sample_unified_records,
    ):
        from grader.src.data.harmonize_labels import LabelHarmonizer

        for source in ["discogs", "ebay_jp"]:
            records = [
                r for r in sample_unified_records if r["source"] == source
            ]
            fname = (
                "discogs_processed.jsonl"
                if source == "discogs"
                else "ebay_processed.jsonl"
            )
            path = tmp_dirs["processed"] / fname
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

        harmonizer = LabelHarmonizer(
            config_path=test_config,
            guidelines_path=guidelines_path,
        )
        records = harmonizer.run(dry_run=True)
        assert len(records) > 0
        for record in records:
            assert "sleeve_label" in record
            assert record["media_label"] != "Generic"


class TestSmokePreprocess:
    def test_preprocess_produces_splits(
        self,
        test_config,
        guidelines_path,
        unified_jsonl_path,
    ):
        from grader.src.data.preprocess import Preprocessor

        preprocessor = Preprocessor(
            config_path=test_config,
            guidelines_path=guidelines_path,
        )
        splits = preprocessor.run(dry_run=True)
        assert set(splits.keys()) == {"train", "val", "test"}
        assert sum(len(v) for v in splits.values()) > 0
        for split_name, records in splits.items():
            for record in records:
                assert "text_clean" in record
                assert record["split"] == split_name


class TestSmokeTFIDF:
    def test_tfidf_produces_sparse_matrices(
        self,
        test_config,
        split_jsonl_paths,
        saved_encoder_paths,
    ):
        import scipy.sparse as sp

        from grader.src.features.tfidf_features import TFIDFFeatureBuilder

        builder = TFIDFFeatureBuilder(config_path=test_config)
        results = builder.run(dry_run=True)

        for split in ["train", "val", "test"]:
            for target in ["sleeve", "media"]:
                X = results["features"][split][target]["X"]
                y = results["features"][split][target]["y"]
                assert sp.issparse(X)
                assert len(y) == X.shape[0]


class TestSmokeBaseline:
    def test_baseline_trains_and_evaluates(
        self,
        test_config,
        saved_encoder_paths,
        saved_feature_paths,
        saved_vectorizer_paths,
        sample_feature_matrices,
        fitted_encoders,
        fitted_baseline,
    ):
        from grader.src.models.baseline import BaselineModel

        baseline = BaselineModel(config_path=test_config)
        baseline.encoders = fitted_encoders
        baseline.models = fitted_baseline
        # Use raw LR as calibrated — avoids class-subset issues
        baseline.calibrated = fitted_baseline

        results = baseline.evaluate(sample_feature_matrices, "test")

        for target in ["sleeve", "media"]:
            assert 0.0 <= results[target]["macro_f1"] <= 1.0
            assert 0.0 <= results[target]["accuracy"] <= 1.0
            assert 0.0 <= results[target]["ece"] <= 1.0


class TestSmokeRuleEngine:
    SEALED_TEXT = "factory sealed, never opened, still in shrink"
    SKIPPING_TEXT = "skipping badly on both sides, unplayable"
    GENERIC_TEXT = "generic white sleeve, die-cut inner only"

    @pytest.fixture
    def engine(self, guidelines_path):
        from grader.src.rules.rule_engine import RuleEngine

        return RuleEngine(guidelines_path=guidelines_path)

    def test_sealed_does_not_trigger_mint_hard_override(
        self, engine, sample_prediction
    ):
        """Mint is model-owned; sealed text alone does not hard-override to Mint."""
        result = engine.apply(sample_prediction, self.SEALED_TEXT)
        assert result["predicted_sleeve_condition"] == "Very Good Plus"
        assert result["metadata"]["rule_override_applied"] is False

    def test_skipping_overrides_to_poor(self, engine, sample_prediction):
        result = engine.apply(sample_prediction, self.SKIPPING_TEXT)
        assert result["predicted_media_condition"] == "Poor"

    def test_generic_overrides_sleeve_not_media(
        self, engine, sample_prediction
    ):
        result = engine.apply(sample_prediction, self.GENERIC_TEXT)
        assert result["predicted_sleeve_condition"] == "Generic"
        assert result["predicted_media_condition"] != "Generic"


class TestSmokeFullInferencePipeline:
    @pytest.fixture
    def pipeline(
        self,
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
        from grader.src.models.baseline import BaselineModel
        from grader.src.pipeline import Pipeline

        pl = Pipeline(config_path=test_config, guidelines_path=guidelines_path)
        pl.infer_model = "baseline"

        bl = BaselineModel(config_path=test_config)
        bl.encoders = fitted_encoders
        bl.models = fitted_calibrated_models
        bl.calibrated = fitted_calibrated_models
        pl._baseline = bl
        return pl

    def test_all_cases_return_valid_schema(self, pipeline):
        for text in [
            "factory sealed, never opened",
            "skipping on side two",
            "untested, sold as seen",
            "sealed record with surface noise",
        ]:
            result = pipeline.predict(text)
            assert "predicted_sleeve_condition" in result
            assert "predicted_media_condition" in result
            assert "confidence_scores" in result
            assert "metadata" in result

    def test_sealed_gives_mint(self, pipeline):
        result = pipeline.predict("factory sealed, still in original shrink")
        assert result["predicted_sleeve_condition"] == "Mint"

    def test_skipping_gives_poor_media(self, pipeline):
        result = pipeline.predict("skipping on side two, badly warped")
        assert result["predicted_media_condition"] == "Poor"

    def test_unverified_flagged_correctly(self, pipeline):
        result = pipeline.predict("untested, sold as seen")
        assert result["metadata"]["media_verifiable"] is False

    def test_contradiction_suppresses_override(self, pipeline):
        result = pipeline.predict("sealed record with heavy surface noise")
        assert result["metadata"]["contradiction_detected"] is True
        assert result["metadata"]["rule_override_applied"] is False

    def test_confidence_scores_sum_to_one(self, pipeline):
        result = pipeline.predict("plays perfectly, light wear")
        sleeve_sum = sum(result["confidence_scores"]["sleeve"].values())
        media_sum = sum(result["confidence_scores"]["media"].values())
        assert sleeve_sum == pytest.approx(1.0, abs=1e-4)
        assert media_sum == pytest.approx(1.0, abs=1e-4)

    def test_batch_predict_consistent_with_single(self, pipeline):
        text = "near mint condition, barely played"
        single = pipeline.predict(text)
        batch = pipeline.predict_batch([text])
        assert (
            single["predicted_sleeve_condition"]
            == batch[0]["predicted_sleeve_condition"]
        )
