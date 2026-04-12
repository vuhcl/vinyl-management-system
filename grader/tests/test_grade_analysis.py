"""
grader/tests/test_grade_analysis.py
"""

import numpy as np
import pytest

from grader.src.evaluation.grade_analysis import (
    build_grade_analysis_report,
    prediction_breakdown_for_class,
    true_label_support,
)


@pytest.fixture
def tiny_classes():
    return np.array(["Excellent", "Near Mint", "Very Good Plus"])


class TestTrueLabelSupport:
    def test_counts(self, tiny_classes):
        y = np.array([0, 0, 1, 2])
        s = true_label_support(y, tiny_classes)
        assert s["Excellent"] == 2
        assert s["Near Mint"] == 1
        assert s["Very Good Plus"] == 1


class TestPredictionBreakdown:
    def test_when_pred_excellent(self, tiny_classes):
        y_true = np.array([1, 1, 0, 2])  # NM, NM, Ex, VG+
        y_pred = np.array([0, 0, 0, 0])  # all predicted Excellent
        n, bd = prediction_breakdown_for_class(
            y_true, y_pred, tiny_classes, "Excellent"
        )
        assert n == 4
        assert bd["Near Mint"] == 2
        assert bd["Excellent"] == 1
        assert bd["Very Good Plus"] == 1


class TestBuildReport:
    def test_contains_support_and_breakdown(self, tiny_classes):
        y_true = np.array([1, 1, 0])
        model = ["Excellent", "Near Mint", "Excellent"]
        rule = ["Excellent", "Near Mint", "Excellent"]
        text = build_grade_analysis_report(
            y_true,
            model,
            rule,
            tiny_classes,
            target="sleeve",
            split="test",
            focus_classes=("Excellent",),
        )
        assert "True label support" in text
        assert "When PREDICTED class = 'Excellent'" in text
        assert "Near Mint" in text
        assert "After rule engine (raw)" in text

    def test_after_for_scoring_adds_effective_block(self, tiny_classes):
        y_true = np.array([1, 1, 0])
        model = ["Excellent", "Near Mint", "Very Good Plus"]
        rule = ["Excellent", "Near Mint", "Excellent"]
        scoring = ["Very Good Plus", "Near Mint", "Very Good Plus"]
        text = build_grade_analysis_report(
            y_true,
            model,
            rule,
            tiny_classes,
            target="sleeve",
            split="test",
            focus_classes=("Excellent",),
            after_for_scoring=scoring,
        )
        assert "Effective for rule-adjusted metrics" in text
        assert "NOTE:" in text

    def test_mismatched_length_raises(self, tiny_classes):
        with pytest.raises(ValueError, match="align"):
            build_grade_analysis_report(
                np.array([0]),
                ["Excellent", "Near Mint"],
                ["Excellent"],
                tiny_classes,
                target="sleeve",
                split="test",
            )
