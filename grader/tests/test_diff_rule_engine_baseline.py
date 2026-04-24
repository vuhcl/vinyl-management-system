"""
Tests for ``grader.src.eval.diff_rule_engine_baseline`` (Track B diff helper).
"""

from __future__ import annotations

from grader.src.eval.diff_rule_engine_baseline import diff_baselines


def _minimal_snapshot(
    delta_f1: float, recall_poor: float, poor_precision: float
) -> dict:
    return {
        "captured_at": "t0",
        "commit": "abc",
        "splits": {
            "test": {
                "media": {
                    "delta_macro_f1": delta_f1,
                    "delta_accuracy": -0.01,
                    "by_after": {
                        "Poor": {
                            "n_helpful": 1,
                            "n_harmful": 2,
                            "n_neutral": 0,
                            "override_precision": poor_precision,
                        }
                    },
                    "slice_recall": {
                        "Poor": {
                            "recall_model": 0.2,
                            "recall_adjusted": recall_poor,
                        }
                    },
                }
            }
        },
    }


def test_diff_baselines_detects_delta() -> None:
    before = _minimal_snapshot(
        delta_f1=-0.02, recall_poor=0.3, poor_precision=0.25
    )
    after = _minimal_snapshot(
        delta_f1=-0.01, recall_poor=0.35, poor_precision=0.30
    )
    text = diff_baselines(before, after, rule_owned=("Poor",))
    assert "delta_macro_f1" in text
    assert "Poor" in text
    assert "by_after[Poor]" in text
    assert "recall_adjusted" in text
    # Same snapshot → no structural crash; output still has headers
    self_diff = diff_baselines(before, before, rule_owned=("Poor",))
    assert "rule_engine_baseline.json diff" in self_diff


def test_diff_baselines_legacy_targets_inverted_shape() -> None:
    """Older JSON used targets[target][split] — still supported."""
    inv = {
        "commit": "x",
        "captured_at": "t",
        "targets": {
            "sleeve": {
                "test": {
                    "delta_macro_f1": 0.0,
                    "delta_accuracy": 0.0,
                    "by_after": {"Poor": {"override_precision": 0.5}},
                    "slice_recall": {
                        "Poor": {"recall_model": 0.1, "recall_adjusted": 0.1}
                    },
                }
            }
        },
    }
    out = diff_baselines(inv, inv, rule_owned=("Poor",))
    assert "[test | sleeve]" in out
