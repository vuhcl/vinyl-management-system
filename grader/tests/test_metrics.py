"""
grader/tests/test_metrics.py
"""

import numpy as np
import pytest

from grader.src.evaluation.metrics import (
    compare_models,
    compute_ece,
    compute_metrics,
    compute_metrics_from_label_strings,
    compute_rule_override_audit,
    log_metrics_to_mlflow,
    remap_true_and_encode_predictions,
    substitute_model_when_pred_excellent,
)


@pytest.fixture
def sleeve_eval_data(fitted_encoders):
    """Synthetic eval data matching sleeve encoder class count."""
    rng = np.random.RandomState(42)
    n = 80
    n_cls = len(fitted_encoders["sleeve"].classes_)

    # Ensure all classes represented
    y_true = np.tile(np.arange(n_cls), n // n_cls + 1)[:n]
    y_pred = y_true.copy()
    noise = rng.choice(n, size=n // 5, replace=False)
    y_pred[noise] = rng.randint(0, n_cls, size=len(noise))

    y_proba = rng.dirichlet(np.ones(n_cls), size=n).astype(np.float32)
    return y_true, y_pred, y_proba


@pytest.fixture
def media_eval_data(fitted_encoders):
    """Synthetic eval data matching media encoder class count."""
    rng = np.random.RandomState(42)
    n = 70
    n_cls = len(fitted_encoders["media"].classes_)

    y_true = np.tile(np.arange(n_cls), n // n_cls + 1)[:n]
    y_pred = y_true.copy()
    noise = rng.choice(n, size=n // 5, replace=False)
    y_pred[noise] = rng.randint(0, n_cls, size=len(noise))

    y_proba = rng.dirichlet(np.ones(n_cls), size=n).astype(np.float32)
    return y_true, y_pred, y_proba


class TestComputeMetrics:
    def test_returns_required_keys(self, sleeve_eval_data, fitted_encoders):
        y_true, y_pred, y_proba = sleeve_eval_data
        result = compute_metrics(
            y_true,
            y_pred,
            y_proba,
            fitted_encoders["sleeve"].classes_,
            target="sleeve",
        )
        for key in ["macro_f1", "accuracy", "ece", "per_class", "class_names"]:
            assert key in result

    def test_macro_f1_between_zero_and_one(
        self, sleeve_eval_data, fitted_encoders
    ):
        y_true, y_pred, y_proba = sleeve_eval_data
        result = compute_metrics(
            y_true,
            y_pred,
            y_proba,
            fitted_encoders["sleeve"].classes_,
            target="sleeve",
        )
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_accuracy_between_zero_and_one(
        self, sleeve_eval_data, fitted_encoders
    ):
        y_true, y_pred, y_proba = sleeve_eval_data
        result = compute_metrics(
            y_true,
            y_pred,
            y_proba,
            fitted_encoders["sleeve"].classes_,
            target="sleeve",
        )
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_perfect_predictions_give_f1_one(self, fitted_encoders):
        n_cls = len(fitted_encoders["sleeve"].classes_)
        n = n_cls * 5
        y_true = np.tile(np.arange(n_cls), 5)
        y_pred = y_true.copy()
        y_proba = np.zeros((n, n_cls))
        for i, label in enumerate(y_true):
            y_proba[i, label] = 1.0

        result = compute_metrics(
            y_true,
            y_pred,
            y_proba,
            fitted_encoders["sleeve"].classes_,
            target="sleeve",
        )
        assert result["macro_f1"] == pytest.approx(1.0, abs=0.01)

    def test_per_class_has_all_grades(self, sleeve_eval_data, fitted_encoders):
        y_true, y_pred, y_proba = sleeve_eval_data
        result = compute_metrics(
            y_true,
            y_pred,
            y_proba,
            fitted_encoders["sleeve"].classes_,
            target="sleeve",
        )
        for cls in fitted_encoders["sleeve"].classes_:
            assert cls in result["per_class"]

    def test_media_target_uses_media_classes(
        self, media_eval_data, fitted_encoders
    ):
        """Media has 7 classes, sleeve has 8 — must not mix them up."""
        y_true, y_pred, y_proba = media_eval_data
        result = compute_metrics(
            y_true,
            y_pred,
            y_proba,
            fitted_encoders["media"].classes_,
            target="media",
        )
        assert len(result["per_class"]) == len(
            fitted_encoders["media"].classes_
        )
        assert "Generic" not in result["per_class"]


class TestComputeECE:
    def test_ece_between_zero_and_one(self, sleeve_eval_data):
        y_true, _, y_proba = sleeve_eval_data
        ece = compute_ece(y_true, y_proba)
        assert 0.0 <= ece <= 1.0

    def test_perfect_calibration_low_ece(self):
        n = 100
        y_true = np.zeros(n, dtype=int)
        y_proba = np.zeros((n, 2))
        y_proba[:, 0] = 1.0
        ece = compute_ece(y_true, y_proba)
        assert ece < 0.05


class TestComputeRuleOverrideAudit:
    def test_helpful_and_harmful_counts(self, fitted_encoders):
        enc = fitted_encoders["sleeve"]
        classes = enc.classes_
        # Two samples: idx 0 and 1
        y_true = np.array([0, 1])
        before = [str(classes[0]), str(classes[1])]
        # First unchanged; second "harmful" if we flip away from true
        after = [str(classes[0]), str(classes[0])]
        aud = compute_rule_override_audit(
            y_true, before, after, enc.classes_, target="sleeve", split="test"
        )
        assert aud["n_changed"] == 1
        assert aud["n_harmful"] == 1
        assert aud["n_helpful"] == 0
        assert aud["override_precision"] == 0.0

    def test_helpful_override(self, fitted_encoders):
        enc = fitted_encoders["sleeve"]
        classes = enc.classes_
        y_true = np.array([1])
        before = [str(classes[0])]
        after = [str(classes[1])]
        aud = compute_rule_override_audit(
            y_true, before, after, enc.classes_, target="sleeve", split="test"
        )
        assert aud["n_helpful"] == 1
        assert aud["n_harmful"] == 0
        assert aud["override_precision"] == 1.0


class TestSubstituteExcellent:
    def test_replaces_excellent_with_model(self):
        before = ["Very Good Plus", "Near Mint", "Excellent"]
        after = ["Excellent", "Near Mint", "Excellent"]
        out = substitute_model_when_pred_excellent(after, before)
        # Rows with rule output Excellent use model label (last row model was already Excellent)
        assert out == ["Very Good Plus", "Near Mint", "Excellent"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            substitute_model_when_pred_excellent(["a"], ["a", "b"])


class TestComputeMetricsFromLabelStrings:
    def test_matches_integer_predictions(self, fitted_encoders):
        """String preds identical to encoded y_true should yield F1 = 1."""
        enc = fitted_encoders["sleeve"]
        n_cls = len(enc.classes_)
        n = n_cls * 3
        y_true = np.tile(np.arange(n_cls), 3)
        labels = [str(enc.classes_[i]) for i in y_true]
        result = compute_metrics_from_label_strings(
            y_true,
            labels,
            enc.classes_,
            target="sleeve",
        )
        assert result["macro_f1"] == pytest.approx(1.0, abs=0.01)
        assert result["accuracy"] == pytest.approx(1.0, abs=0.01)
        assert result["ece"] is None

    def test_pred_label_missing_from_encoder_expands_space(
        self, fitted_encoders
    ):
        """Predictions may use grades absent from an older encoder (e.g. Excellent)."""
        enc = fitted_encoders["sleeve"]
        y_true = np.array([0])
        r = compute_metrics_from_label_strings(
            y_true,
            ["NotARealGrade"],
            enc.classes_,
            target="sleeve",
        )
        assert "NotARealGrade" in r["class_names"]


class TestRemapTrueAndEncode:
    def test_adds_excellent_to_space(self):
        class_names = np.array(["Near Mint", "Very Good"])
        y_true = np.array([0, 1, 0])
        y_rem, combined, (yp,) = remap_true_and_encode_predictions(
            y_true, class_names, ["Excellent", "Very Good", "Near Mint"]
        )
        assert "Excellent" in combined
        assert y_rem.shape == y_true.shape
        assert len(yp) == 3


class TestCompareModels:
    def test_compare_models_returns_string(
        self, sleeve_eval_data, media_eval_data, fitted_encoders
    ):
        y_true_s, y_pred_s, y_proba_s = sleeve_eval_data
        y_true_m, y_pred_m, y_proba_m = media_eval_data

        base_metrics = {
            "sleeve": compute_metrics(
                y_true_s,
                y_pred_s,
                y_proba_s,
                fitted_encoders["sleeve"].classes_,
                "sleeve",
            ),
            "media": compute_metrics(
                y_true_m,
                y_pred_m,
                y_proba_m,
                fitted_encoders["media"].classes_,
                "media",
            ),
        }
        result = compare_models(base_metrics, base_metrics)
        assert isinstance(result, str)
        assert "Baseline" in result
        assert "Transformer" in result

    def test_positive_improvement_shown(
        self, sleeve_eval_data, fitted_encoders
    ):
        n_cls = len(fitted_encoders["sleeve"].classes_)
        n = n_cls * 5
        y_true = np.tile(np.arange(n_cls), 5)
        y_proba = np.eye(n_cls)[y_true % n_cls]

        rng = np.random.RandomState(0)
        y_bad = rng.randint(0, n_cls, size=n)

        base_metrics = {
            "sleeve": compute_metrics(
                y_true,
                y_bad,
                y_proba,
                fitted_encoders["sleeve"].classes_,
                "sleeve",
            ),
        }
        better_metrics = {
            "sleeve": compute_metrics(
                y_true,
                y_true,
                y_proba,
                fitted_encoders["sleeve"].classes_,
                "sleeve",
            ),
        }
        result = compare_models(base_metrics, better_metrics)
        assert "✓" in result
