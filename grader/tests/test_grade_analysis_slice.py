"""
grader/tests/test_grade_analysis_slice.py

Pytest coverage for the Track A rule-owned slice helpers and the
stratified override audit (see
``.cursor/plans/rule-owned_eval_reporting_afdea43a.plan.md`` §2–§4).

These tests use small in-memory arrays rather than the full pipeline,
so they stay fast and do not depend on a trained model. They assert
exact counts, so they also serve as regression fixtures for the
``by_after`` / ``by_transition`` breakdown and the TXT report that
the pipeline appends to ``grade_analysis_{split}.txt``.
"""

from __future__ import annotations

import numpy as np
import pytest

from grader.src.evaluation.grade_analysis import (
    build_rule_owned_slice_report,
    resolve_rule_owned_grades,
    slice_precision_for_grade,
    slice_recall_for_grade,
    true_label_breakdown_for_grade,
)
from grader.src.evaluation.metrics import (
    compute_rule_override_audit,
    format_override_audit_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SLEEVE_CLASSES = np.array(
    [
        "Mint",
        "Near Mint",
        "Excellent",
        "Very Good Plus",
        "Very Good",
        "Good",
        "Poor",
        "Generic",
    ]
)
MEDIA_CLASSES = np.array(
    [
        "Mint",
        "Near Mint",
        "Excellent",
        "Very Good Plus",
        "Very Good",
        "Good",
        "Poor",
    ]
)


def _encode(labels: list[str], class_names: np.ndarray) -> np.ndarray:
    idx = {str(c): i for i, c in enumerate(class_names)}
    return np.array([idx[str(l)] for l in labels], dtype=int)


# ---------------------------------------------------------------------------
# resolve_rule_owned_grades
# ---------------------------------------------------------------------------
class TestResolveRuleOwnedGrades:
    def test_sleeve_and_media_ownership(self, guidelines_path):
        import yaml

        with open(guidelines_path) as f:
            guidelines = yaml.safe_load(f)
        mapping = resolve_rule_owned_grades(guidelines)

        # Poor + Generic on sleeve, Poor only on media
        assert set(mapping["sleeve"]) == {"Poor", "Generic"}
        assert set(mapping["media"]) == {"Poor"}

    def test_empty_on_missing_owners(self):
        mapping = resolve_rule_owned_grades({"grades": {}, "grade_owners": {}})
        assert mapping == {"sleeve": [], "media": []}


# ---------------------------------------------------------------------------
# true_label_breakdown_for_grade / slice_recall / slice_precision
# ---------------------------------------------------------------------------
class TestSliceHelpers:
    # 4 rows: 3 true-Poor (1 correct, 2 wrong), 1 true-VG predicted Poor.
    def _fixture_media(self):
        y_true_labels = ["Poor", "Poor", "Poor", "Very Good"]
        y_pred_labels = ["Poor", "Very Good", "Good", "Poor"]
        y_true = _encode(y_true_labels, MEDIA_CLASSES)
        y_pred = _encode(y_pred_labels, MEDIA_CLASSES)
        return y_true, y_pred

    def test_true_label_breakdown_counts(self):
        y_true, y_pred = self._fixture_media()
        n, bd = true_label_breakdown_for_grade(
            y_true, y_pred, MEDIA_CLASSES, "Poor"
        )
        assert n == 3
        assert bd == {"Poor": 1, "Very Good": 1, "Good": 1}

    def test_slice_recall(self):
        y_true, y_pred = self._fixture_media()
        rec = slice_recall_for_grade(y_true, y_pred, MEDIA_CLASSES, "Poor")
        # 1 correct out of 3 true-Poor rows.
        assert rec == pytest.approx(1 / 3, abs=1e-4)

    def test_slice_precision(self):
        y_true, y_pred = self._fixture_media()
        prec = slice_precision_for_grade(y_true, y_pred, MEDIA_CLASSES, "Poor")
        # 2 rows predicted Poor; 1 is truly Poor.
        assert prec == pytest.approx(0.5, abs=1e-4)

    def test_unsupported_grade_returns_none(self):
        y_true, y_pred = self._fixture_media()
        assert (
            slice_recall_for_grade(y_true, y_pred, MEDIA_CLASSES, "Mint")
            is None
        )
        assert (
            slice_precision_for_grade(
                y_true, y_pred, MEDIA_CLASSES, "Excellent"
            )
            is None
        )


# ---------------------------------------------------------------------------
# build_rule_owned_slice_report — text structure
# ---------------------------------------------------------------------------
class TestRuleOwnedSliceReport:
    def test_report_contains_banner_and_grades(self):
        y_true = _encode(
            ["Poor", "Poor", "Very Good", "Very Good Plus"], MEDIA_CLASSES
        )
        y_pred_model = ["Very Good", "Very Good", "Very Good", "Very Good Plus"]
        y_pred_rule = ["Poor", "Very Good", "Very Good", "Very Good Plus"]
        text = build_rule_owned_slice_report(
            y_true=y_true,
            y_pred_model=y_pred_model,
            y_pred_rule=y_pred_rule,
            class_names=MEDIA_CLASSES,
            target="media",
            split="test",
            rule_owned_grades=["Poor"],
        )
        assert "RULE-OWNED SLICE" in text
        assert "target=MEDIA" in text
        assert "split=test" in text
        assert "True label = 'Poor'" in text
        assert "Slice recall (model only" in text
        assert "Slice precision" in text
        assert "Predicted histogram" in text

    def test_report_no_support_line(self):
        # No rows with true==Generic — section should render "(no support)".
        y_true = _encode(["Poor", "Very Good"], MEDIA_CLASSES)
        text = build_rule_owned_slice_report(
            y_true=y_true,
            y_pred_model=["Poor", "Very Good"],
            y_pred_rule=["Poor", "Very Good"],
            class_names=MEDIA_CLASSES,
            target="media",
            split="val",
            rule_owned_grades=["Poor", "Generic"],
        )
        assert "True label = 'Generic'" in text
        assert "(no support)" in text or "(grade not in merged" in text


# ---------------------------------------------------------------------------
# compute_rule_override_audit — stratified outputs
# ---------------------------------------------------------------------------
class TestStratifiedOverrideAudit:
    def test_by_after_and_by_transition(self):
        """
        Fixture with 3 overrides:
          row 0: before=Very Good, after=Poor, true=Poor       → helpful
          row 1: before=Very Good, after=Poor, true=Very Good  → harmful
          row 2: before=Good,      after=Poor, true=Near Mint  → neutral
          row 3: no change (Very Good Plus).
        Expected:
          by_after["Poor"] = {n_changed=3, helpful=1, harmful=1, neutral=1,
                              override_precision=0.5}
          by_transition["Very Good->Poor"] has n_changed=2,
                         helpful=1, harmful=1 → precision=0.5.
        """
        y_true_labels = ["Poor", "Very Good", "Near Mint", "Very Good Plus"]
        y_before = ["Very Good", "Very Good", "Good", "Very Good Plus"]
        y_after = ["Poor", "Poor", "Poor", "Very Good Plus"]

        y_true = _encode(y_true_labels, MEDIA_CLASSES)
        audit = compute_rule_override_audit(
            y_true=y_true,
            y_pred_before=y_before,
            y_pred_after=y_after,
            class_names=MEDIA_CLASSES,
            target="media",
            split="test",
        )

        assert audit["n_changed"] == 3
        assert audit["n_helpful"] == 1
        assert audit["n_harmful"] == 1
        assert audit["n_neutral"] == 1
        assert audit["override_precision"] == pytest.approx(0.5, abs=1e-4)

        by_after = audit["by_after"]
        assert set(by_after.keys()) == {"Poor"}
        assert by_after["Poor"]["n_changed"] == 3
        assert by_after["Poor"]["n_helpful"] == 1
        assert by_after["Poor"]["n_harmful"] == 1
        assert by_after["Poor"]["n_neutral"] == 1
        assert by_after["Poor"]["override_precision"] == pytest.approx(
            0.5, abs=1e-4
        )

        by_tr = audit["by_transition"]
        assert "Very Good->Poor" in by_tr
        tr = by_tr["Very Good->Poor"]
        assert tr["n_changed"] == 2
        assert tr["n_helpful"] == 1
        assert tr["n_harmful"] == 1
        assert tr["override_precision"] == pytest.approx(0.5, abs=1e-4)

    def test_format_report_renders_by_after(self):
        y_true = _encode(["Poor", "Very Good"], MEDIA_CLASSES)
        audit = compute_rule_override_audit(
            y_true=y_true,
            y_pred_before=["Very Good", "Very Good"],
            y_pred_after=["Poor", "Poor"],
            class_names=MEDIA_CLASSES,
            target="media",
            split="test",
        )
        text = format_override_audit_report(audit)
        assert "RULE OVERRIDE AUDIT" in text
        assert "target=MEDIA" in text
        assert "By final predicted grade" in text
        # The single override destination must render in the by-after table.
        assert "Poor" in text


# ---------------------------------------------------------------------------
# Track B regression fixtures — static guideline edits
# ---------------------------------------------------------------------------
class TestRegressionGuidelineFixtures:
    """
    Positive + adversarial fixtures pinned to the §13b/§14b edits:
      - Poor strict media signals still fire on single match.
      - Poor cosignal (``skipping``) requires corroboration.
      - Poor forbidden softeners block catastrophic misfires.
      - Generic strict phrases fire alone; ambiguous ``white sleeve``
        requires corroboration.
      - Generic forbiddens block override when an original cover is
        explicitly mentioned.
      - NM ``forbidden_exceptions`` un-block disclaimed tape/stickers.
    """

    @pytest.fixture
    def engine(self, guidelines_path):
        from grader.src.rules.rule_engine import RuleEngine

        return RuleEngine(guidelines_path=guidelines_path)

    # --- Poor media: strict vs cosignal ------------------------------
    def test_poor_strict_media_fires_alone(self, engine):
        assert (
            engine.check_hard_override("record is completely unplayable", "media")
            == "Poor"
        )

    def test_poor_single_skip_fires_alone(self, engine):
        # Per user guidance: any skip is below Good's "plays through"
        # baseline — strict Poor, no corroboration needed.
        assert (
            engine.check_hard_override("just one skip on side a", "media")
            == "Poor"
        )

    def test_poor_plural_skips_fire_alone(self, engine):
        assert (
            engine.check_hard_override(
                "a few skips on side b, otherwise plays", "media"
            )
            == "Poor"
        )

    def test_poor_skipping_strict_still_fires_with_other_signal(
        self, engine
    ):
        assert (
            engine.check_hard_override(
                "skipping and won't play at all", "media"
            )
            == "Poor"
        )

    def test_poor_turntable_dependent_skip_does_not_fire(self, engine):
        # Inconsistent / setup-dependent skip — the only legitimate
        # non-Poor skip case, blocked via ``forbidden_signals``.
        assert (
            engine.check_hard_override(
                "skip depending on the turntable and tonearm weight",
                "media",
            )
            is None
        )

    def test_poor_negated_skipping_does_not_fire(self, engine):
        # ``no skipping`` negation must still block Poor.
        assert (
            engine.check_hard_override(
                "plays cleanly, no skipping across either side",
                "media",
            )
            is None
        )

    def test_poor_not_warped_disclaimer_blocks(self, engine):
        # "not warped" is the explicit disclaimer — must block any
        # spurious ``badly warped`` misfire via negation confusion.
        assert (
            engine.check_hard_override(
                "checked thoroughly, not warped and plays cleanly",
                "media",
            )
            is None
        )

    # --- Poor false-negative patterns (§10, missed phrasings) --------
    def test_poor_will_not_play_fires_alone(self, engine):
        # Near-synonym of the existing ``won't play`` strict.
        assert (
            engine.check_hard_override(
                "record simply will not play", "media"
            )
            == "Poor"
        )

    def test_poor_severely_warped_fires_alone(self, engine):
        assert (
            engine.check_hard_override("disc is severely warped", "media")
            == "Poor"
        )

    def test_poor_skips_throughout_fires_alone(self, engine):
        # "throughout" qualifier elevates plain "skips" to catastrophic.
        assert (
            engine.check_hard_override(
                "skips throughout the entire side", "media"
            )
            == "Poor"
        )

    def test_poor_falling_apart_fires_alone_on_sleeve(self, engine):
        assert (
            engine.check_hard_override(
                "jacket is literally falling apart", "sleeve"
            )
            == "Poor"
        )

    def test_poor_falling_apart_not_fired_on_media(self, engine):
        # Sleeve-only hard signal — must not fire on media.
        assert (
            engine.check_hard_override(
                "the paper lyric sheet is falling apart", "media"
            )
            is None
        )

    def test_poor_cannot_play_cosignal_needs_corroboration(self, engine):
        # "cannot play" is demoted to cosignal — a lone mention (e.g. a
        # speed-specific or format-specific limitation) must not fire.
        assert (
            engine.check_hard_override(
                "cannot play on my old auto-changer at 45 rpm", "media"
            )
            is None
        )

    def test_poor_cannot_play_with_corroboration_fires(self, engine):
        assert (
            engine.check_hard_override(
                "cannot play — disc is cracked at the edge", "media"
            )
            == "Poor"
        )

    def test_poor_plays_great_contradiction_suppresses_override(
        self, engine
    ):
        # Self-contradictory text ("skipping" + "plays great") routes
        # through the contradictions layer (suppresses all overrides,
        # model decides). The hard-override call alone would still
        # return Poor because ``skipping`` is strict; contradiction
        # suppression happens in ``apply`` at the pipeline level.
        assert engine.check_contradiction(
            "skipping reported but plays great on my setup"
        )

    # --- Poor sleeve -------------------------------------------------
    def test_poor_sleeve_strict_fires_alone(self, engine):
        assert (
            engine.check_hard_override(
                "cover is fully split along the spine", "sleeve"
            )
            == "Poor"
        )

    def test_poor_sleeve_cosignal_water_damage_needs_corroboration(self, engine):
        # Lone ``heavy water damage`` is cosignal — must not fire.
        assert (
            engine.check_hard_override(
                "jacket has heavy water damage throughout", "sleeve"
            )
            is None
        )

    def test_poor_media_strict_not_triggered_on_sleeve(self, engine):
        # ``deep gouges`` is now media-only per §13b — must NOT fire Poor
        # when applied to the sleeve target.
        assert (
            engine.check_hard_override("deep gouges on the disc", "sleeve")
            is None
        )

    # --- Generic -----------------------------------------------------
    def test_generic_strict_fires_alone(self, engine):
        assert (
            engine.check_hard_override("this is a generic sleeve", "sleeve")
            == "Generic"
        )

    def test_generic_cosignal_white_sleeve_alone_does_not_fire(self, engine):
        # "white sleeve" is cosignal; without another Generic cue it must
        # not fire (grammatically ambiguous — could be a white *inner*).
        assert (
            engine.check_hard_override("comes in a white sleeve", "sleeve")
            is None
        )

    def test_generic_with_original_cover_forbidden_blocks(self, engine):
        # Even though ``generic sleeve`` matches (strict), the forbidden
        # ``original cover`` must block the Generic override — seller is
        # saying the jacket is there and also a company inner sleeve.
        assert (
            engine.check_hard_override(
                "has the original cover and a generic sleeve inside", "sleeve"
            )
            is None
        )

    def test_generic_forbidden_exception_no_original_cover(self, engine):
        # ``no original cover`` is both a strict Generic signal AND a
        # ``forbidden_exceptions_sleeve`` entry — the exception strip
        # keeps ``original cover`` from blocking this legitimate Generic.
        assert (
            engine.check_hard_override(
                "no original cover — ships in a plain white sleeve",
                "sleeve",
            )
            == "Generic"
        )

    # --- Generic cosignal demotions (§11) ---------------------------
    def test_generic_promo_sleeve_alone_does_not_fire(self, engine):
        # ``promo sleeve`` was demoted from strict to cosignal per §11
        # because promos can ship with their original jacket.
        assert (
            engine.check_hard_override(
                "promo sleeve, comes with original cover too", "sleeve"
            )
            is None
        )

    def test_generic_promo_sleeve_with_corroboration_fires(self, engine):
        # Corroborated by an unambiguous ``no original cover`` strict.
        assert (
            engine.check_hard_override(
                "promo sleeve, no original cover", "sleeve"
            )
            == "Generic"
        )

    def test_generic_inner_sleeve_only_alone_does_not_fire(self, engine):
        # Demoted to cosignal — "inner sleeve only" sometimes annotates
        # a condition sub-note rather than a housing statement.
        assert (
            engine.check_hard_override(
                "inner sleeve only has minor wear", "sleeve"
            )
            is None
        )

    # --- Generic new strict entries (§10/§14b) ----------------------
    def test_generic_replacement_sleeve_fires_alone(self, engine):
        assert (
            engine.check_hard_override(
                "housed in a replacement sleeve", "sleeve"
            )
            == "Generic"
        )

    def test_generic_generic_jacket_fires_alone(self, engine):
        # ``generic jacket`` mirrors ``generic sleeve`` / ``generic cover``.
        assert (
            engine.check_hard_override("ships in a generic jacket", "sleeve")
            == "Generic"
        )

    # --- Generic new forbiddens (§14b) ------------------------------
    def test_generic_picture_sleeve_forbidden_blocks(self, engine):
        # ``picture sleeve`` is a dedicated 7" cover — not a generic
        # housing. A strict match elsewhere must be blocked.
        assert (
            engine.check_hard_override(
                "comes in its original picture sleeve, generic sleeve inside",
                "sleeve",
            )
            is None
        )

    def test_generic_obi_forbidden_blocks(self, engine):
        # ``obi`` indicates a complete JP issue with original housing.
        assert (
            engine.check_hard_override(
                "includes obi — generic sleeve inside the outer", "sleeve"
            )
            is None
        )

    def test_generic_with_sleeve_forbidden_blocks(self, engine):
        # ``with sleeve`` asserts a housing is present. A strict Generic
        # cue (``no cover``) fires the hard match but the forbidden
        # layer must block once ``with sleeve`` appears.
        assert (
            engine.check_hard_override(
                "no cover — comes with sleeve and insert", "sleeve"
            )
            is None
        )

    # --- Generic new forbidden_exceptions (§11) ---------------------
    def test_generic_promo_sleeve_with_cover_exception_unblocks(self, engine):
        # ``promo sleeve with cover`` is the exception phrase — it
        # strips the ``with cover`` forbidden so a corroborated Generic
        # still fires when the seller uses that benign boilerplate.
        # Without the exception, ``with cover`` would block the hard
        # override below.
        assert (
            engine.check_hard_override(
                "promo sleeve with cover, no original cover inside",
                "sleeve",
            )
            == "Generic"
        )

    # --- Generic false-negative patterns (§11, missed phrasings) ----
    def test_generic_stock_sleeve_fires_alone(self, engine):
        assert (
            engine.check_hard_override(
                "ships in a stock sleeve from the distributor", "sleeve"
            )
            == "Generic"
        )

    def test_generic_no_jacket_fires_alone(self, engine):
        assert (
            engine.check_hard_override(
                "record only — no jacket, no insert", "sleeve"
            )
            == "Generic"
        )

    def test_generic_jacket_missing_fires_alone(self, engine):
        # Note: "original jacket missing" is blocked because the
        # forbidden ``original jacket`` fires first; use a phrasing
        # without the word "original" so ``jacket missing`` strict
        # fires cleanly.
        assert (
            engine.check_hard_override(
                "jacket missing — record only in plastic", "sleeve"
            )
            == "Generic"
        )

    def test_generic_paper_sleeve_alone_does_not_fire(self, engine):
        # "paper sleeve" is ambiguous: could be the inner paper of an
        # intact jacket. Cosignal requires corroboration.
        assert (
            engine.check_hard_override(
                "stored in its original paper sleeve", "sleeve"
            )
            is None
        )

    def test_generic_cardboard_sleeve_with_corroboration_fires(self, engine):
        # Cardboard sleeve (cosignal) + no original cover (strict).
        assert (
            engine.check_hard_override(
                "cardboard sleeve, no original cover", "sleeve"
            )
            == "Generic"
        )

    def test_generic_cardboard_sleeve_alone_does_not_fire(self, engine):
        # "cardboard sleeve" alone could describe an intact cardboard
        # jacket — the forbidden ``original cover`` in the same text
        # would block anyway, but verify the cosignal logic first.
        assert (
            engine.check_hard_override(
                "cardboard sleeve in great shape", "sleeve"
            )
            is None
        )

    # --- Contradictions expansion (§12) ------------------------------
    def test_contradiction_plays_perfectly_cracked(self, engine):
        # "plays perfectly" + "cracked" contradict — signals the row
        # should have its rule overrides suppressed upstream.
        assert engine.check_contradiction(
            "plays perfectly, cracked along the edge"
        )

    def test_contradiction_sealed_cracked(self, engine):
        assert engine.check_contradiction(
            "still sealed — supposedly cracked per prior listing"
        )

    def test_contradiction_plays_fine_skipping(self, engine):
        # "plays fine" + "skipping" — blocks the cosignal corroboration
        # path when a seller is reporting someone else's review.
        assert engine.check_contradiction(
            "plays fine on my deck, previous owner said skipping"
        )

    def test_contradiction_never_played_cracked(self, engine):
        assert engine.check_contradiction(
            "never played, claim of cracked is mistaken"
        )

    # --- NM forbidden_exceptions ------------------------------------
    def test_nm_disclaimed_tape_does_not_block_soft_override(self, engine):
        """
        ``tape`` is a NM forbidden, but ``no tape`` / ``without tape``
        were added to ``forbidden_exceptions_sleeve`` in §14b. A NM
        candidate with "no tape, no stickers" must still be eligible.
        """
        grade = engine.check_soft_override(
            text="no marks, no defects, no tape, no stickers",
            target="sleeve",
            model_confidence=0.40,
            predicted_grade="Excellent",
        )
        assert grade == "Near Mint"
