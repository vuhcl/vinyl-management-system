"""
grader/tests/test_metrics.py
"""

import numpy as np
import pytest

from grader.src.evaluation.metrics import (
    compare_models,
    compute_ece,
    compute_metrics,
    log_metrics_to_mlflow,
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
