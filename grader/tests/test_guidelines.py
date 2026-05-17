"""
grader/tests/test_guidelines.py

Tests for grading_guidelines.yaml schema validation.
Ensures all required keys exist, grade lists are consistent,
and eBay JP harmonization covers all expected grades.
"""

import pytest
import yaml

from grader.src.guidelines_identity import (
    guidelines_version_from_mapping,
    is_valid_guidelines_version_format,
)


@pytest.fixture
def guidelines(guidelines_path):
    with open(guidelines_path) as f:
        return yaml.safe_load(f)


class TestGuidelinesVersion:
    def test_version_present(self, guidelines):
        assert "guidelines_version" in guidelines

    def test_version_format(self, guidelines):
        v = guidelines_version_from_mapping(guidelines)
        assert is_valid_guidelines_version_format(v)


class TestCanonicalGrades:
    def test_sleeve_grades_present(self, guidelines):
        assert "sleeve_grades" in guidelines

    def test_media_grades_present(self, guidelines):
        assert "media_grades" in guidelines

    def test_sleeve_has_generic(self, guidelines):
        assert "Generic" in guidelines["sleeve_grades"]

    def test_media_has_no_generic(self, guidelines):
        assert "Generic" not in guidelines["media_grades"]

    def test_sleeve_has_seven_condition_grades(self, guidelines):
        # 7 condition grades + Generic = 8 sleeve grades
        condition_grades = [
            g for g in guidelines["sleeve_grades"] if g != "Generic"
        ]
        assert len(condition_grades) == 7

    def test_media_has_seven_grades(self, guidelines):
        assert len(guidelines["media_grades"]) == 7

    def test_grade_ordinal_map_excludes_generic(self, guidelines):
        assert "Generic" not in guidelines["grade_ordinal_map"]

    def test_grade_ordinal_map_covers_all_condition_grades(self, guidelines):
        ordinal = guidelines["grade_ordinal_map"]
        for grade in guidelines["media_grades"]:
            assert grade in ordinal, f"{grade} missing from grade_ordinal_map"

    def test_ordinal_map_is_sequential(self, guidelines):
        values = sorted(guidelines["grade_ordinal_map"].values())
        assert values == list(range(len(values)))

    def test_mint_is_best(self, guidelines):
        ordinal = guidelines["grade_ordinal_map"]
        assert ordinal["Mint"] == min(ordinal.values())

    def test_poor_is_worst(self, guidelines):
        ordinal = guidelines["grade_ordinal_map"]
        assert ordinal["Poor"] == max(ordinal.values())


class TestGradeOwnership:
    def test_grade_owners_present(self, guidelines):
        assert "grade_owners" in guidelines

    def test_mint_owned_by_model(self, guidelines):
        """Mint hard-override removed — model owns Mint predictions."""
        assert guidelines["grade_owners"]["Mint"] == "model"

    def test_poor_owned_by_rule_engine(self, guidelines):
        assert guidelines["grade_owners"]["Poor"] == "rule_engine"

    def test_generic_owned_by_rule_engine(self, guidelines):
        assert guidelines["grade_owners"]["Generic"] == "rule_engine"

    def test_model_grades_owned_by_model(self, guidelines):
        model_grades = [
            "Near Mint", "Excellent", "Very Good Plus", "Very Good", "Good"
        ]
        for grade in model_grades:
            assert guidelines["grade_owners"][grade] == "model"


class TestEbayJPHarmonization:
    EXPECTED_EBAY_GRADES = ["S", "M-", "E+", "E", "E-", "VG+", "VG"]

    def test_harmonization_present(self, guidelines):
        assert "ebay_jp_harmonization" in guidelines

    def test_all_expected_grades_present(self, guidelines):
        harmonization = guidelines["ebay_jp_harmonization"]
        for grade in self.EXPECTED_EBAY_GRADES:
            assert grade in harmonization, f"eBay grade {grade} missing"

    def test_s_maps_to_mint(self, guidelines):
        assert guidelines["ebay_jp_harmonization"]["S"]["canonical"] == "Mint"

    def test_vgplus_maps_to_very_good_plus(self, guidelines):
        assert (
            guidelines["ebay_jp_harmonization"]["VG+"]["canonical"]
            == "Very Good Plus"
        )

    def test_vg_maps_to_very_good(self, guidelines):
        assert (
            guidelines["ebay_jp_harmonization"]["VG"]["canonical"]
            == "Very Good"
        )

    def test_e_and_eminus_map_to_excellent(self, guidelines):
        assert guidelines["ebay_jp_harmonization"]["E"]["canonical"]  == "Excellent"
        assert guidelines["ebay_jp_harmonization"]["E-"]["canonical"] == "Excellent"

    def test_eminus_is_flagged(self, guidelines):
        assert guidelines["ebay_jp_harmonization"]["E-"]["flagged"] is True

    def test_all_have_label_confidence(self, guidelines):
        for grade, entry in guidelines["ebay_jp_harmonization"].items():
            assert "label_confidence" in entry, f"{grade} missing label_confidence"
            assert 0.0 <= entry["label_confidence"] <= 1.0

    def test_canonical_grades_are_valid(self, guidelines):
        valid = set(guidelines["sleeve_grades"]) | set(guidelines["media_grades"])
        for grade, entry in guidelines["ebay_jp_harmonization"].items():
            assert entry["canonical"] in valid, (
                f"{grade} maps to unknown canonical grade: {entry['canonical']}"
            )


class TestGradeDefinitions:
    REQUIRED_KEYS = [
        "applies_to",
        "description",
        "rule_confidence_threshold",
    ]

    # Hard-signal schema allows three forms (any suffices):
    #   1. Legacy ``hard_signals: [...]``
    #   2. Untargeted ``hard_signals_strict`` / ``hard_signals_cosignal``
    #   3. Per-target ``hard_signals_{strict,cosignal}_{sleeve,media}``
    # Empty legacy ``hard_signals: []`` on non-rule-owned grades is still
    # valid as a placeholder.
    @staticmethod
    def _has_hard_signal_schema(grade_def: dict) -> bool:
        legacy = grade_def.get("hard_signals")
        if isinstance(legacy, list):
            return True
        targeted = any(
            isinstance(grade_def.get(k), list)
            for k in (
                "hard_signals_strict",
                "hard_signals_cosignal",
                "hard_signals_strict_sleeve",
                "hard_signals_strict_media",
                "hard_signals_cosignal_sleeve",
                "hard_signals_cosignal_media",
            )
        )
        return targeted

    @staticmethod
    def _has_supporting_schema(grade_def: dict) -> bool:
        if grade_def.get("supporting_signals") is not None:
            return True
        return (
            grade_def.get("supporting_signals_sleeve") is not None
            and grade_def.get("supporting_signals_media") is not None
        )

    @staticmethod
    def _has_forbidden_schema(grade_def: dict) -> bool:
        if grade_def.get("forbidden_signals") is not None:
            return True
        return (
            grade_def.get("forbidden_signals_sleeve") is not None
            and grade_def.get("forbidden_signals_media") is not None
        )

    @staticmethod
    def _has_min_supporting_schema(grade_def: dict) -> bool:
        if grade_def.get("min_supporting") is not None:
            return True
        return (
            grade_def.get("min_supporting_sleeve") is not None
            and grade_def.get("min_supporting_media") is not None
        )

    def test_all_grades_defined(self, guidelines):
        expected = set(guidelines["sleeve_grades"])
        defined  = set(guidelines["grades"].keys())
        assert expected == defined

    def test_required_keys_present(self, guidelines):
        for grade, grade_def in guidelines["grades"].items():
            for key in self.REQUIRED_KEYS:
                assert key in grade_def, (
                    f"Grade '{grade}' missing required key: {key}"
                )
            assert self._has_hard_signal_schema(grade_def), (
                f"Grade '{grade}' needs hard_signals or "
                "hard_signals_strict[_sleeve|_media] / "
                "hard_signals_cosignal[_sleeve|_media]"
            )
            assert self._has_supporting_schema(grade_def), (
                f"Grade '{grade}' needs supporting_signals or "
                "supporting_signals_sleeve + supporting_signals_media"
            )
            assert self._has_forbidden_schema(grade_def), (
                f"Grade '{grade}' needs forbidden_signals or "
                "forbidden_signals_sleeve + forbidden_signals_media"
            )
            assert self._has_min_supporting_schema(grade_def), (
                f"Grade '{grade}' needs min_supporting or "
                "min_supporting_sleeve + min_supporting_media"
            )

    def test_max_supporting_sane_when_present(self, guidelines):
        for grade, grade_def in guidelines["grades"].items():
            if "max_supporting" in grade_def:
                mx = grade_def["max_supporting"]
                mn = grade_def["min_supporting"]
                assert isinstance(mx, int), f"{grade}: max_supporting must be int"
                assert mn is not None
                assert mx >= mn, (
                    f"{grade}: max_supporting ({mx}) must be >= min_supporting ({mn})"
                )
            for target in ("sleeve", "media"):
                mx_key = f"max_supporting_{target}"
                mn_key = f"min_supporting_{target}"
                if mx_key not in grade_def:
                    continue
                mx = grade_def[mx_key]
                mn = grade_def.get(mn_key)
                assert mn is not None, (
                    f"{grade}: {mx_key} requires matching {mn_key}"
                )
                assert isinstance(mx, int), f"{grade}: {mx_key} must be int"
                assert mx >= mn, (
                    f"{grade}: {mx_key} ({mx}) must be >= {mn_key} ({mn})"
                )

    def test_applies_to_valid_targets(self, guidelines):
        valid_targets = {"sleeve", "media"}
        for grade, grade_def in guidelines["grades"].items():
            for target in grade_def["applies_to"]:
                assert target in valid_targets

    def test_generic_applies_to_sleeve_only(self, guidelines):
        assert guidelines["grades"]["Generic"]["applies_to"] == ["sleeve"]

    def test_mint_has_hard_signals(self, guidelines):
        # Mint is model-owned; its ``hard_signals`` list is retained as
        # curated lexicon for future feature-builder work (documented in
        # the YAML). The rule engine's ``check_hard_override`` does not
        # evaluate Mint.
        assert len(guidelines["grades"]["Mint"]["hard_signals"]) > 0

    def test_poor_has_hard_signals(self, guidelines):
        # Poor uses per-target strict + cosignal keys after the §13b
        # migration — the legacy ``hard_signals`` list is intentionally
        # removed so there is a single source of truth per target.
        poor = guidelines["grades"]["Poor"]
        assert (
            len(poor.get("hard_signals_strict_media", [])) > 0
            or len(poor.get("hard_signals_strict_sleeve", [])) > 0
        ), "Poor must declare at least one strict hard signal per target"

    def test_generic_has_hard_signals(self, guidelines):
        # Generic is sleeve-only; uses untargeted strict/cosignal keys.
        generic = guidelines["grades"]["Generic"]
        assert len(generic.get("hard_signals_strict", [])) > 0

    def test_poor_rule_confidence_is_one(self, guidelines):
        assert guidelines["grades"]["Poor"]["rule_confidence_threshold"] == 1.0

    def test_generic_rule_confidence_is_one(self, guidelines):
        assert guidelines["grades"]["Generic"]["rule_confidence_threshold"] == 1.0

    def test_sealed_in_mint_hard_signals(self, guidelines):
        assert "sealed" in guidelines["grades"]["Mint"]["hard_signals"]


class TestContradictions:
    def test_contradictions_present(self, guidelines):
        assert "contradictions" in guidelines

    def test_contradictions_are_pairs(self, guidelines):
        for pair in guidelines["contradictions"]:
            assert len(pair) == 2

    def test_sealed_surface_noise_contradiction(self, guidelines):
        pairs = [tuple(p) for p in guidelines["contradictions"]]
        assert ("sealed", "surface noise") in pairs

    def test_plays_perfectly_skipping_contradiction(self, guidelines):
        pairs = [tuple(p) for p in guidelines["contradictions"]]
        assert ("plays perfectly", "skipping") in pairs


class TestDiscogsConditionMap:
    EXPECTED_DISCOGS_CONDITIONS = [
        "Mint (M)",
        "Near Mint (NM or M-)",
        "Very Good Plus (VG+)",
        "Very Good (VG)",
        "Good Plus (G+)",
        "Good (G)",
        "Fair (F)",
        "Poor (P)",
        "Generic Sleeve",
    ]

    def test_condition_map_present(self, guidelines):
        assert "discogs_condition_map" in guidelines

    def test_all_discogs_conditions_mapped(self, guidelines):
        condition_map = guidelines["discogs_condition_map"]
        for condition in self.EXPECTED_DISCOGS_CONDITIONS:
            assert condition in condition_map, (
                f"Discogs condition '{condition}' not in condition map"
            )

    def test_generic_sleeve_maps_to_generic(self, guidelines):
        assert (
            guidelines["discogs_condition_map"]["Generic Sleeve"] == "Generic"
        )

    def test_good_and_good_plus_both_map_to_good(self, guidelines):
        cmap = guidelines["discogs_condition_map"]
        assert cmap["Good (G)"]      == "Good"
        assert cmap["Good Plus (G+)"] == "Good"

    def test_fair_and_poor_both_map_to_poor(self, guidelines):
        cmap = guidelines["discogs_condition_map"]
        assert cmap["Fair (F)"] == "Poor"
        assert cmap["Poor (P)"] == "Poor"
