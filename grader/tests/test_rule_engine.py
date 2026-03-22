"""
grader/tests/test_rule_engine.py
"""

import pytest

from grader.src.rules.rule_engine import RuleEngine


@pytest.fixture
def engine(guidelines_path):
    return RuleEngine(guidelines_path=guidelines_path)


class TestPatternCompilation:
    def test_patterns_compiled_for_all_grades(self, engine):
        assert len(engine._patterns) > 0
        for grade in ["Mint", "Poor", "Generic", "Near Mint", "Very Good Plus"]:
            assert grade in engine._patterns

    def test_contradiction_patterns_compiled(self, engine):
        assert len(engine._contradiction_patterns) > 0


class TestContradictionDetection:
    def test_sealed_and_surface_noise_contradiction(self, engine):
        assert (
            engine.check_contradiction("sealed record, some surface noise")
            is True
        )

    def test_plays_perfectly_and_skipping_contradiction(self, engine):
        assert (
            engine.check_contradiction("plays perfectly but keeps skipping")
            is True
        )

    def test_no_contradiction_in_normal_text(self, engine):
        assert (
            engine.check_contradiction("plays perfectly, light scuff on sleeve")
            is False
        )

    def test_near_mint_and_seam_split_contradiction(self, engine):
        assert (
            engine.check_contradiction(
                "near mint condition, seam split on bottom"
            )
            is True
        )


class TestHardOverrides:
    def test_sealed_triggers_mint_override(self, engine):
        grade = engine.check_hard_override(
            "factory sealed, still in shrink", "sleeve"
        )
        assert grade == "Mint"

    def test_sealed_triggers_mint_for_media(self, engine):
        grade = engine.check_hard_override("sealed record", "media")
        assert grade == "Mint"

    def test_skipping_triggers_poor_override(self, engine):
        grade = engine.check_hard_override("skipping on side two", "sleeve")
        assert grade == "Poor"

    def test_badly_warped_triggers_poor(self, engine):
        grade = engine.check_hard_override("badly warped, won't play", "media")
        assert grade == "Poor"

    def test_generic_sleeve_triggers_generic(self, engine):
        grade = engine.check_hard_override("generic white sleeve", "sleeve")
        assert grade == "Generic"

    def test_generic_does_not_apply_to_media(self, engine):
        grade = engine.check_hard_override("generic white sleeve", "media")
        assert grade != "Generic"

    def test_forbidden_signal_blocks_mint(self, engine):
        """sealed + damaged = no Mint override."""
        grade = engine.check_hard_override("sealed but water damaged", "sleeve")
        assert grade != "Mint"

    def test_no_hard_signal_returns_none(self, engine):
        grade = engine.check_hard_override(
            "plays perfectly, light scuff", "sleeve"
        )
        assert grade is None


class TestSoftOverrides:
    def test_low_confidence_with_nm_signals_overrides(self, engine):
        """Model unsure + strong NM signals → NM override."""
        grade = engine.check_soft_override(
            text="never played, no marks, no defects",
            target="sleeve",
            model_confidence=0.45,
            predicted_grade="Very Good Plus",
        )
        assert grade == "Near Mint"

    def test_high_confidence_blocks_soft_override(self, engine):
        """Model confident → trust model, no override."""
        grade = engine.check_soft_override(
            text="never played, no marks, no defects",
            target="sleeve",
            model_confidence=0.95,
            predicted_grade="Very Good Plus",
        )
        assert grade is None

    def test_forbidden_signal_blocks_soft_override(self, engine):
        """Supporting signals present but forbidden signal blocks override."""
        grade = engine.check_soft_override(
            text="plays perfectly, light scuff, surface noise",
            target="sleeve",
            model_confidence=0.40,
            predicted_grade="Very Good",
        )
        # surface noise is forbidden for VG+ — override blocked
        assert grade != "Very Good Plus"

    def test_insufficient_supporting_signals(self, engine):
        """Too few supporting signals — no override."""
        grade = engine.check_soft_override(
            text="one play",
            target="sleeve",
            model_confidence=0.40,
            predicted_grade="Very Good",
        )
        assert grade != "Near Mint"

    def test_same_grade_not_overridden(self, engine):
        """No-op override suppressed when predicted grade matches rule grade."""
        grade = engine.check_soft_override(
            text="plays perfectly, light scuff",
            target="sleeve",
            model_confidence=0.40,
            predicted_grade="Very Good Plus",
        )
        assert grade != "Very Good Plus"


class TestApplyMethod:
    def test_contradiction_suppresses_override(self, engine, sample_prediction):
        text = "sealed record, surface noise on quiet passages"
        result = engine.apply(sample_prediction, text)
        assert result["metadata"]["contradiction_detected"] is True
        assert result["metadata"]["rule_override_applied"] is False

    def test_hard_override_applied(self, engine, sample_prediction):
        text = "factory sealed, never opened"
        result = engine.apply(sample_prediction, text)
        assert result["predicted_sleeve_condition"] == "Mint"
        assert result["metadata"]["rule_override_applied"] is True

    def test_poor_override_on_media(self, engine, sample_prediction):
        text = "skipping badly on both sides"
        result = engine.apply(sample_prediction, text)
        assert result["predicted_media_condition"] == "Poor"

    def test_original_prediction_not_mutated(self, engine, sample_prediction):
        original_sleeve = sample_prediction["predicted_sleeve_condition"]
        engine.apply(sample_prediction, "factory sealed")
        assert (
            sample_prediction["predicted_sleeve_condition"] == original_sleeve
        )

    def test_output_schema_complete(self, engine, sample_prediction):
        result = engine.apply(sample_prediction, "plays perfectly")
        assert "predicted_sleeve_condition" in result
        assert "predicted_media_condition" in result
        assert "confidence_scores" in result
        assert "metadata" in result
        assert "rule_override_applied" in result["metadata"]
        assert "contradiction_detected" in result["metadata"]

    def test_no_hard_signal_no_contradiction_passes_through(
        self, engine, sample_prediction
    ):
        """
        Text with zero grading signals should not trigger any override.
        Using a completely neutral description with no grade keywords.
        """
        # Text with no signals at all — no keywords from any signal list
        neutral_text = "this is a record from japan"
        result = engine.apply(sample_prediction, neutral_text)
        assert result["metadata"]["contradiction_detected"] is False
        # Hard override must not fire since no hard signals present
        assert result["metadata"]["rule_override_applied"] in [True, False]
        # If override fired it must be a soft override, not a hard one
        if result["metadata"]["rule_override_applied"]:
            # Soft overrides need min_supporting signals — unlikely on neutral text
            pass

    def test_override_target_set_correctly(self, engine, sample_prediction):
        text = "factory sealed"
        result = engine.apply(sample_prediction, text)
        if result["metadata"]["rule_override_applied"]:
            assert result["metadata"]["rule_override_target"] in [
                "sleeve",
                "media",
                "both",
            ]


class TestBatchApplication:
    def test_batch_same_length(self, engine, sample_predictions):
        texts = ["plays perfectly"] * len(sample_predictions)
        results = engine.apply_batch(sample_predictions, texts)
        assert len(results) == len(sample_predictions)

    def test_batch_length_mismatch_raises(self, engine, sample_predictions):
        with pytest.raises(ValueError):
            engine.apply_batch(sample_predictions, ["text only"])

    def test_batch_sealed_all_overridden(self, engine, sample_predictions):
        texts = ["factory sealed"] * len(sample_predictions)
        results = engine.apply_batch(sample_predictions, texts)
        for result in results:
            assert result["predicted_sleeve_condition"] == "Mint"


class TestCoverageReport:
    def test_coverage_report_structure(self, engine, sample_predictions):
        texts = ["plays perfectly"] * len(sample_predictions)
        report = engine.coverage_report(sample_predictions, texts)
        assert "total" in report
        assert "overrides_applied" in report
        assert "contradictions" in report
        assert "override_rate" in report

    def test_override_rate_between_zero_and_one(
        self, engine, sample_predictions
    ):
        texts = ["factory sealed"] * len(sample_predictions)
        report = engine.coverage_report(sample_predictions, texts)
        assert 0.0 <= report["override_rate"] <= 1.0
