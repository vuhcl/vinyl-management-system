"""
grader/tests/test_baseline.py
"""

import numpy as np
import pytest

from grader.src.models.baseline import BaselineModel


@pytest.fixture
def baseline(
    test_config,
    saved_encoder_paths,
    saved_feature_paths,
    saved_vectorizer_paths,
):
    return BaselineModel(config_path=test_config)


@pytest.fixture
def trained_baseline(
    baseline, sample_feature_matrices, fitted_encoders, fitted_baseline
):
    """
    Baseline with fitted models. Uses raw LR as calibrated model
    to avoid class-subset issues with CalibratedClassifierCV on
    small synthetic datasets.
    """
    baseline.encoders = fitted_encoders
    baseline.models = fitted_baseline
    # Use raw LR as calibrated — sufficient for schema/wiring tests
    baseline.calibrated = fitted_baseline
    return baseline


class TestLoadTrainedFromArtifacts:
    def test_load_eval_from_disk(
        self,
        test_config,
        saved_encoder_paths,
        saved_vectorizer_paths,
        saved_feature_paths,
        saved_calibrated_model_paths,
    ):
        from grader.src.models.baseline import BaselineModel

        bl, bundle = BaselineModel.load_trained_from_artifacts(test_config)
        assert "eval" in bundle
        assert "test" in bundle["eval"]
        for target in ("sleeve", "media"):
            assert target in bl.calibrated
            assert bundle["eval"]["test"][target]["macro_f1"] >= 0.0


class TestTraining:
    def test_train_returns_both_heads(
        self, baseline, sample_feature_matrices, fitted_encoders
    ):
        baseline.encoders = fitted_encoders
        models = baseline.train(sample_feature_matrices)
        assert "sleeve" in models
        assert "media" in models

    def test_both_heads_fitted(
        self, baseline, sample_feature_matrices, fitted_encoders
    ):
        baseline.encoders = fitted_encoders
        models = baseline.train(sample_feature_matrices)
        for target, model in models.items():
            assert hasattr(model, "coef_")


class TestCalibration:
    def test_calibration_returns_calibrated_classifiers(
        self,
        baseline,
        sample_feature_matrices,
        fitted_encoders,
        fitted_baseline,
    ):
        """
        Tests that calibrate() runs without error.
        Uses train features to ensure all classes are seen
        — val split may not contain all grades in small fixtures.
        """
        baseline.encoders = fitted_encoders
        baseline.models = fitted_baseline
        # Train as val so every grade appears (tiny val split may omit classes).
        cal_features = {
            "val": sample_feature_matrices["train"],
            "train": sample_feature_matrices["train"],
            "test": sample_feature_matrices["test"],
        }
        calibrated = baseline.calibrate(baseline.models, cal_features)
        assert "sleeve" in calibrated
        assert "media" in calibrated

    def test_calibration_preserves_lr_coefficients(
        self,
        baseline,
        sample_feature_matrices,
        fitted_encoders,
        fitted_baseline,
    ):
        """Frozen base: logistic regression weights must not change on val."""
        baseline.encoders = fitted_encoders
        baseline.models = fitted_baseline
        coef_before = {
            t: np.asarray(m.coef_.copy()) for t, m in fitted_baseline.items()
        }
        cal_features = {
            "val": sample_feature_matrices["train"],
            "train": sample_feature_matrices["train"],
            "test": sample_feature_matrices["test"],
        }
        calibrated = baseline.calibrate(baseline.models, cal_features)
        for target in ("sleeve", "media"):
            inner = calibrated[target].calibrated_classifiers_[0].estimator
            lr = inner.estimator if hasattr(inner, "estimator") else inner
            np.testing.assert_allclose(
                np.asarray(lr.coef_),
                coef_before[target],
                rtol=0,
                atol=0,
            )

    def test_calibrated_probas_sum_to_one(
        self,
        baseline,
        sample_feature_matrices,
        fitted_encoders,
        fitted_baseline,
    ):
        baseline.encoders = fitted_encoders
        baseline.models = fitted_baseline
        baseline.calibrated = fitted_baseline  # raw LR has valid predict_proba

        X = sample_feature_matrices["test"]["sleeve"]["X"]
        proba = baseline.calibrated["sleeve"].predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-5)


class TestPrediction:
    def test_predict_returns_correct_count(
        self, trained_baseline, sample_feature_matrices
    ):
        X_sleeve = sample_feature_matrices["test"]["sleeve"]["X"]
        X_media = sample_feature_matrices["test"]["media"]["X"]
        results = trained_baseline.predict(X_sleeve, X_media)
        assert len(results) == X_sleeve.shape[0]

    def test_prediction_schema_complete(
        self, trained_baseline, sample_feature_matrices
    ):
        X_sleeve = sample_feature_matrices["test"]["sleeve"]["X"]
        X_media = sample_feature_matrices["test"]["media"]["X"]
        result = trained_baseline.predict(X_sleeve, X_media)[0]

        assert "item_id" in result
        assert "predicted_sleeve_condition" in result
        assert "predicted_media_condition" in result
        assert "confidence_scores" in result
        assert "sleeve" in result["confidence_scores"]
        assert "media" in result["confidence_scores"]
        assert "metadata" in result

    def test_confidence_scores_sum_to_one(
        self, trained_baseline, sample_feature_matrices
    ):
        X_sleeve = sample_feature_matrices["test"]["sleeve"]["X"]
        X_media = sample_feature_matrices["test"]["media"]["X"]
        result = trained_baseline.predict(X_sleeve, X_media)[0]

        sleeve_sum = sum(result["confidence_scores"]["sleeve"].values())
        media_sum = sum(result["confidence_scores"]["media"].values())
        assert sleeve_sum == pytest.approx(1.0, abs=1e-4)
        assert media_sum == pytest.approx(1.0, abs=1e-4)

    def test_predicted_grade_is_valid_canonical(
        self, trained_baseline, sample_feature_matrices, guidelines_path
    ):
        import yaml

        guidelines = yaml.safe_load(open(guidelines_path))
        sleeve_grades = set(guidelines["sleeve_grades"])
        media_grades = set(guidelines["media_grades"])

        X_sleeve = sample_feature_matrices["test"]["sleeve"]["X"]
        X_media = sample_feature_matrices["test"]["media"]["X"]
        results = trained_baseline.predict(X_sleeve, X_media)

        for result in results:
            assert result["predicted_sleeve_condition"] in sleeve_grades
            assert result["predicted_media_condition"] in media_grades

    def test_rule_override_false_at_model_level(
        self, trained_baseline, sample_feature_matrices
    ):
        X_sleeve = sample_feature_matrices["test"]["sleeve"]["X"]
        X_media = sample_feature_matrices["test"]["media"]["X"]
        results = trained_baseline.predict(X_sleeve, X_media)
        for result in results:
            assert result["metadata"]["rule_override_applied"] is False


class TestEvaluation:
    def test_evaluate_returns_metrics_for_both_targets(
        self, trained_baseline, sample_feature_matrices
    ):
        results = trained_baseline.evaluate(sample_feature_matrices, "test")
        assert "sleeve" in results
        assert "media" in results

    def test_metrics_keys_present(
        self, trained_baseline, sample_feature_matrices
    ):
        results = trained_baseline.evaluate(sample_feature_matrices, "test")
        for target in ["sleeve", "media"]:
            assert "macro_f1" in results[target]
            assert "accuracy" in results[target]
            assert "ece" in results[target]

    def test_macro_f1_in_range(self, trained_baseline, sample_feature_matrices):
        results = trained_baseline.evaluate(sample_feature_matrices, "test")
        for target in ["sleeve", "media"]:
            assert 0.0 <= results[target]["macro_f1"] <= 1.0

    def test_ece_in_range(self, trained_baseline, sample_feature_matrices):
        results = trained_baseline.evaluate(sample_feature_matrices, "test")
        for target in ["sleeve", "media"]:
            assert 0.0 <= results[target]["ece"] <= 1.0
