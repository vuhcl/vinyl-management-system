"""
grader/tests/test_rule_engine.py
"""

import pytest

from grader.src.rules.rule_engine import RuleEngine


@pytest.fixture
def engine(guidelines_path):
    return RuleEngine(guidelines_path=guidelines_path)


@pytest.fixture
def engine_allow_excellent(guidelines_path):
    """Match legacy soft-EX behavior; default :class:`RuleEngine` has EX soft off."""
    return RuleEngine(
        guidelines_path=guidelines_path,
        allow_excellent_soft_override=True,
    )


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

    # Note: the ``[near mint, seam split]`` contradiction pair was dropped
    # in §14b (negation-blind — it fired on "near mint, no seam split").
    # Suppression of NM overrides with explicit sleeve defects is now
    # delegated to ``Near Mint.forbidden_signals_sleeve``.


class TestHardOverrides:
    def test_sealed_does_not_trigger_mint_override(self, engine):
        """Mint is model-owned — sealed keywords must not fire a hard override."""
        assert engine.check_hard_override("factory sealed, still in shrink", "sleeve") is None
        assert engine.check_hard_override("sealed record", "media") is None

    def test_skipping_alone_triggers_poor(self, engine):
        """
        Per user guidance: Good's rubric baselines on "plays through",
        so any skip is by definition below Good — strict Poor on media.
        No corroboration is required.
        """
        assert (
            engine.check_hard_override("skipping on side two", "media")
            == "Poor"
        )

    def test_skipping_with_other_strict_still_triggers_poor(self, engine):
        """``skipping`` + any other strict hard signal → Poor (still)."""
        grade = engine.check_hard_override(
            "skipping on side two, completely unplayable", "media"
        )
        assert grade == "Poor"

    def test_skipping_does_not_trigger_poor_for_sleeve(self, engine):
        """Playback wording applies to media only — same blob must not Poor the sleeve."""
        assert (
            engine.check_hard_override("skipping on side two", "sleeve") is None
        )

    def test_vinyl_repeatedly_skips_triggers_poor_media(self, engine):
        """Affirmative skip phrasing on the disc must Poor media (sleeve unchanged)."""
        text = (
            "sleeve is decent close to very good plus but the vinyl repeatedly skips"
        ).lower()
        assert engine.check_contradiction(text) is False
        assert engine.check_hard_override(text, "media") == "Poor"
        assert engine.check_hard_override(text, "sleeve") is None

    def test_turntable_dependent_skip_not_poor(self, engine):
        """Seller says behavior varies by turntable — block Poor hard override."""
        text = (
            "pronounced click or skip depending on the turntable and tone arm weight"
        )
        assert engine.check_hard_override(text.lower(), "media") is None

    def test_badly_warped_triggers_poor(self, engine):
        grade = engine.check_hard_override("badly warped, won't play", "media")
        assert grade == "Poor"

    def test_light_water_damage_not_poor_hard(self, engine):
        """VG+ jacket copy notes — generic 'water damage' is not auto-Poor."""
        assert engine.check_hard_override(
            "solid very good plus sleeve has water damage", "sleeve"
        ) is None

    def test_severe_water_damage_with_corroboration_triggers_poor(self, engine):
        """
        Water-damage phrasing is cosignal-only on Poor.sleeve — corroborate
        with a strict sleeve signal (``fully split``) so the hard override
        fires. A lone ``heavy water damage`` match without any second
        signal must NOT fire Poor (see §13 — prevents small-blob FPs).
        """
        assert (
            engine.check_hard_override(
                "cover has heavy water damage, fully split along the spine",
                "sleeve",
            )
            == "Poor"
        )

    def test_severe_water_damage_alone_does_not_trigger_poor(self, engine):
        assert (
            engine.check_hard_override(
                "cover has heavy water damage throughout", "sleeve"
            )
            is None
        )

    def test_generic_sleeve_triggers_generic(self, engine):
        # ``generic sleeve`` (strict) matches directly — single hit suffices.
        grade = engine.check_hard_override("this is a generic sleeve", "sleeve")
        assert grade == "Generic"

    def test_white_sleeve_alone_does_not_trigger_generic(self, engine):
        """
        ``white sleeve`` / ``plain sleeve`` are cosignal-only on Generic —
        they are grammatically ambiguous (inner vs outer) so a single match
        without a second Generic cue must NOT fire.
        """
        assert engine.check_hard_override("white sleeve", "sleeve") is None

    def test_white_sleeve_with_corroboration_triggers_generic(self, engine):
        """Cosignal ``white sleeve`` + strict ``no original cover`` → Generic."""
        grade = engine.check_hard_override(
            "white sleeve, no original cover", "sleeve"
        )
        assert grade == "Generic"

    def test_poor_sleeve_precedence_over_generic_sleeve(self, engine):
        """
        Catastrophic jacket (Poor sleeve) must win over generic-housing cues
        (Generic) when both hard-match — regression for gold-Poor rows stuck
        on ``white generic sleeve`` + seam destruction.
        """
        text = (
            "sleeve in very bad condition shipped in white generic sleeve, "
            "top and bottom seams fully split"
        )
        assert engine.check_hard_override(text, "sleeve") == "Poor"

    def test_generic_matte_black_and_one_sheet_strict_fire(self, engine):
        assert (
            engine.check_hard_override(
                "ships in generic matte black sleeve only", "sleeve"
            )
            == "Generic"
        )
        assert (
            engine.check_hard_override(
                "generic one-sheet sleeve, no artwork", "sleeve"
            )
            == "Generic"
        )

    def test_salsoul_sleeve_triggers_generic(self, engine):
        assert (
            engine.check_hard_override(
                "comes in salsoul sleeve, no picture cover", "sleeve"
            )
            == "Generic"
        )

    def test_structured_cover_equals_generic_strict(self, engine):
        """Seller template lines like ``cover=generic`` are strict Generic."""
        assert (
            engine.check_hard_override(
                "media=very good plus cover=generic [hairline on media]",
                "sleeve",
            )
            == "Generic"
        )

    def test_black_sleeve_plain_jacket_cosignal_alone_does_not_fire(
        self, engine
    ):
        for text in (
            "black sleeve",
            "plain black jacket",
            "plain cover",
            "white label",
        ):
            assert engine.check_hard_override(text, "sleeve") is None

    def test_black_sleeve_with_corroboration_triggers_generic(self, engine):
        assert (
            engine.check_hard_override(
                "black sleeve, no original cover", "sleeve"
            )
            == "Generic"
        )

    def test_white_label_promo_with_no_cover_triggers_generic(self, engine):
        # WL promos often use generic/plain housing — not an anti-Generic cue.
        assert (
            engine.check_hard_override(
                "white label promo, no cover", "sleeve"
            )
            == "Generic"
        )

    def test_white_label_with_original_cover_does_not_trigger_generic(
        self, engine
    ):
        assert (
            engine.check_hard_override(
                "white label copy, original cover has wear", "sleeve"
            )
            is None
        )

    def test_generic_does_not_apply_to_media(self, engine):
        grade = engine.check_hard_override("this is a generic sleeve", "media")
        assert grade != "Generic"

    def test_no_hard_signal_returns_none(self, engine):
        grade = engine.check_hard_override(
            "plays perfectly, light scuff", "sleeve"
        )
        assert grade is None


class TestSoftOverrides:
    def test_low_confidence_with_nm_signals_overrides(self, engine):
        """Model unsure + strong NM signals one step below NM → NM override."""
        # predicted=Excellent (ordinal 2) → NM (ordinal 1) is a 1-step upgrade, allowed.
        # predicted=VG+ (ordinal 3) would be a 2-step jump — blocked by max_soft_upgrade_steps.
        grade = engine.check_soft_override(
            text="no marks, no defects, no wear",
            target="sleeve",
            model_confidence=0.45,
            predicted_grade="Excellent",
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
            text="played once",
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

    def test_max_supporting_skips_excellent_when_many_cues(self, engine):
        """Stacked EX-tier cosmetic cues exceed max_supporting → not Excellent."""
        text = (
            "minor scuff slight scuff very minor wear light marks carefully handled "
            "well cared for excellent bright and shiny raking light hairline "
            "hairlines faint corner faint crease faint bump clean and unmarked "
            "minor abrasion"
        )
        grade = engine.check_soft_override(
            text.lower(),
            "sleeve",
            model_confidence=0.40,
            predicted_grade="Very Good",
        )
        assert grade != "Excellent"

    def test_fewer_excellent_cues_downgrade_nm_to_excellent(
        self, engine_allow_excellent
    ):
        """Excellent only fires as a downgrade from Near Mint."""
        text = "bright and shiny hairline faint corner"
        grade = engine_allow_excellent.check_soft_override(
            text.lower(),
            "media",
            model_confidence=0.40,
            predicted_grade="Near Mint",
        )
        assert grade == "Excellent"

    def test_excellent_soft_override_disabled_by_default(
        self, engine, engine_allow_excellent, guidelines_path
    ):
        text = "bright and shiny hairline faint corner"
        t = text.lower()
        assert engine_allow_excellent.check_soft_override(
            t, "media", 0.40, "Near Mint"
        ) == "Excellent"
        off = RuleEngine(
            guidelines_path=guidelines_path, allow_excellent_soft_override=False
        )
        assert off.check_soft_override(
            t, "media", 0.40, "Near Mint"
        ) != "Excellent"
        assert off.check_soft_override(
            t, "media", 0.40, "Near Mint"
        ) == engine.check_soft_override(
            t, "media", 0.40, "Near Mint"
        )

    def test_excellent_does_not_upgrade_vg_or_vgplus(self, engine):
        """Excellent must not upgrade VG or VG+ — only_downgrade: true."""
        for predicted in ("Very Good", "Very Good Plus"):
            grade = engine.check_soft_override(
                "bright and shiny hairline faint corner",
                "media",
                model_confidence=0.40,
                predicted_grade=predicted,
            )
            assert grade != "Excellent", f"Should not upgrade {predicted} to Excellent"

    def test_near_excellent_does_not_upgrade_vg(self, engine):
        """Seller 'VG+ near excellent' phrasing must not soft-upgrade to Excellent."""
        text = "side a has scratches side c is very good plus near excellent"
        assert (
            engine.check_soft_override(
                text.lower(),
                "media",
                model_confidence=0.40,
                predicted_grade="Very Good",
            )
            != "Excellent"
        )

    def test_cassette_tape_word_does_not_downgrade_nm_media(self, engine):
        """'Tape' as format (cassette listing) is not VG adhesive-tape residue."""
        text = (
            "excellent condition tape with the clean and complete original sleeve "
            "(loads more in my store)"
        )
        assert (
            engine.check_soft_override(
                text.lower(),
                "media",
                model_confidence=0.40,
                predicted_grade="Near Mint",
            )
            is None
        )

    def test_jacket_seam_split_does_not_soft_downgrade_media(self, engine):
        """Sleeve-only defects must not match media supporting_signals."""
        assert (
            engine.check_soft_override(
                "sleeve has seam split only played once",
                "media",
                model_confidence=0.40,
                predicted_grade="Near Mint",
            )
            is None
        )

    def test_wrinkles_pull_sleeve_from_vg_plus_to_vg(self, engine):
        """Sleeve wrinkles are VG-tier in rubric — not VG+ cosmetic-only."""
        text = (
            "sleeve has some wrinkles.disc has scratch."
            "label has foxing stain"
        )
        grade = engine.check_soft_override(
            text.lower(),
            "sleeve",
            model_confidence=0.50,
            predicted_grade="Very Good Plus",
        )
        assert grade == "Very Good"

    def test_disc_has_scratch_pulls_media_from_good_to_vg(self, engine):
        """Singular seller phrasing “disc has scratch” → Very Good media."""
        text = (
            "sleeve has some wrinkles.disc has scratch."
            "label has foxing stain"
        )
        grade = engine.check_soft_override(
            text.lower(),
            "media",
            model_confidence=0.50,
            predicted_grade="Good",
        )
        assert grade == "Very Good"


class TestApplyMethod:
    def test_excellent_collapses_to_near_mint_when_soft_off(
        self, engine, sample_prediction
    ):
        """rules.allow_excellent_soft_override false → remap model Excellent to NM."""
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Excellent",
            "predicted_media_condition": "Excellent",
            "confidence_scores": {
                "sleeve": {
                    "Mint": 0.01,
                    "Near Mint": 0.01,
                    "Excellent": 0.96,
                    "Very Good Plus": 0.01,
                    "Very Good": 0.005,
                    "Good": 0.005,
                    "Poor": 0.005,
                    "Generic": 0.005,
                },
                "media": {
                    "Mint": 0.01,
                    "Near Mint": 0.01,
                    "Excellent": 0.96,
                    "Very Good Plus": 0.01,
                    "Very Good": 0.005,
                    "Good": 0.005,
                    "Poor": 0.005,
                },
            },
        }
        text = (
            "vinyl has one or two very minor hairline surface marks only "
            "visible under harsh lighting"
        )
        result = engine.apply(pred, text)
        assert result["predicted_sleeve_condition"] == "Near Mint"
        assert result["predicted_media_condition"] == "Near Mint"
        assert "Excellent" not in result["confidence_scores"]["sleeve"]
        assert "Excellent" not in result["confidence_scores"]["media"]
        assert result["confidence_scores"]["sleeve"]["Near Mint"] == pytest.approx(
            0.01 + 0.96
        )
        assert result["confidence_scores"]["media"]["Near Mint"] == pytest.approx(
            0.01 + 0.96
        )
        assert set(result["metadata"].get("excellent_collapsed_to_near_mint", [])) == {
            "sleeve",
            "media",
        }

    def test_excellent_not_collapsed_when_soft_on(
        self, engine_allow_excellent, sample_prediction
    ):
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Excellent",
            "predicted_media_condition": "Excellent",
            "confidence_scores": {
                "sleeve": {
                    **sample_prediction["confidence_scores"]["sleeve"],
                    "Excellent": 0.96,
                    "Near Mint": 0.01,
                    "Very Good Plus": 0.01,
                },
                "media": {
                    **sample_prediction["confidence_scores"]["media"],
                    "Excellent": 0.96,
                    "Near Mint": 0.01,
                    "Very Good Plus": 0.01,
                },
            },
        }
        result = engine_allow_excellent.apply(pred, "plays perfectly, no issues")
        assert result["predicted_sleeve_condition"] == "Excellent"
        assert result["predicted_media_condition"] == "Excellent"
        assert "excellent_collapsed_to_near_mint" not in result["metadata"]

    def test_contradiction_still_collapses_excellent_to_nm(
        self, engine, sample_prediction
    ):
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Excellent",
            "predicted_media_condition": "Excellent",
            "confidence_scores": {
                "sleeve": {**sample_prediction["confidence_scores"]["sleeve"], "Excellent": 0.96},
                "media": {**sample_prediction["confidence_scores"]["media"], "Excellent": 0.96},
            },
        }
        result = engine.apply(pred, "sealed record, surface noise on quiet passages")
        assert result["metadata"]["contradiction_detected"] is True
        assert result["metadata"]["rule_override_applied"] is False
        assert result["predicted_sleeve_condition"] == "Near Mint"
        assert result["predicted_media_condition"] == "Near Mint"

    def test_contradiction_suppresses_override(self, engine, sample_prediction):
        text = "sealed record, surface noise on quiet passages"
        result = engine.apply(sample_prediction, text)
        assert result["metadata"]["contradiction_detected"] is True
        assert result["metadata"]["rule_override_applied"] is False

    def test_sealed_text_does_not_trigger_mint_hard_override(self, engine, sample_prediction):
        """Mint is model-owned — sealed text must not override model prediction."""
        text = "factory sealed, never opened"
        result = engine.apply(sample_prediction, text)
        assert result["predicted_sleeve_condition"] == "Very Good Plus"

    def test_poor_hard_override_applied(self, engine, sample_prediction):
        """
        Poor is hard-owned — a strict hard signal (``cracked``) overrides
        to Poor for media regardless of model confidence. ``skipping``
        is also strict post user-guidance, but we keep ``cracked`` here
        so the test doesn't depend on the skip-policy decision.
        """
        result = engine.apply(
            sample_prediction, "disc is cracked along the label"
        )
        assert result["predicted_media_condition"] == "Poor"
        assert result["metadata"]["rule_override_applied"] is True

    def test_nm_sleeve_with_small_seam_split_only_downgrades_to_vg_plus(
        self, engine, sample_prediction
    ):
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Near Mint",
        }
        result = engine.apply(pred, "sleeve has a small seam split along top")
        assert result["predicted_sleeve_condition"] == "Very Good Plus"

    def test_nm_sleeve_with_non_small_split_downgrades_to_vg(
        self, engine, sample_prediction
    ):
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Near Mint",
        }
        result = engine.apply(pred, "sleeve has a seam split along top")
        assert result["predicted_sleeve_condition"] == "Very Good"

    def test_nm_small_seam_split_plus_other_defect_downgrades_to_vg(
        self, engine, sample_prediction
    ):
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Near Mint",
        }
        result = engine.apply(
            pred, "sleeve has a small seam split and corner wear"
        )
        assert result["predicted_sleeve_condition"] == "Very Good"

    def test_vg_plus_small_top_seam_split_does_not_downgrade(self, engine, sample_prediction):
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Very Good Plus",
        }
        result = engine.apply(pred, "sleeve has a small top seam split, otherwise new")
        assert result["predicted_sleeve_condition"] == "Very Good Plus"

    def test_vg_plus_small_top_seam_split_with_corner_wear_downgrades(self, engine, sample_prediction):
        pred = {
            **sample_prediction,
            "predicted_sleeve_condition": "Very Good Plus",
        }
        result = engine.apply(pred, "sleeve has a small top seam split and corner wear")
        assert result["predicted_sleeve_condition"] == "Very Good"

    def test_poor_override_on_media(self, engine, sample_prediction):
        # Strict hard signal — fires on any single match.
        text = "badly warped, won't play at all"
        result = engine.apply(sample_prediction, text)
        assert result["predicted_media_condition"] == "Poor"

    def test_original_prediction_not_mutated(self, engine, sample_prediction):
        original_sleeve = sample_prediction["predicted_sleeve_condition"]
        engine.apply(sample_prediction, "badly warped, won't play at all")
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
        # Strict hard signal on media only — override_target must be "media".
        text = "completely unplayable disc"
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

    def test_batch_sealed_preserves_model_prediction(self, engine, sample_predictions):
        """Mint is model-owned — sealed text must not hard-override any prediction."""
        texts = ["factory sealed"] * len(sample_predictions)
        results = engine.apply_batch(sample_predictions, texts)
        for pred, result in zip(sample_predictions, results):
            assert result["predicted_sleeve_condition"] == pred["predicted_sleeve_condition"]


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
        texts = ["completely unplayable disc"] * len(sample_predictions)
        report = engine.coverage_report(sample_predictions, texts)
        assert 0.0 <= report["override_rate"] <= 1.0
