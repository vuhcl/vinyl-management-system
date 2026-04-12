"""
grader/tests/test_preprocess.py

Tests for preprocess.py — text normalization, abbreviation expansion,
protected term preservation, unverified media detection, and
adaptive stratified splitting.
"""

import re

import pytest

from grader.src.data.preprocess import Preprocessor


@pytest.fixture
def preprocessor(test_config, guidelines_path):
    return Preprocessor(
        config_path=test_config,
        guidelines_path=guidelines_path,
    )


class TestAbbreviationExpansion:
    def test_nm_expands(self, preprocessor):
        result = preprocessor.clean_text("NM sleeve")
        assert "near mint" in result

    def test_vgplus_expands(self, preprocessor):
        result = preprocessor.clean_text("VG+ record")
        assert "very good plus" in result

    def test_vgplusplus_expands_correctly(self, preprocessor):
        """vg++ must expand to very good plus, not corrupt vg+."""
        result = preprocessor.clean_text("VG++ sleeve")
        assert "very good plus" in result
        assert "++" not in result

    def test_vgplusplus_before_vgplus(self, preprocessor):
        """
        Critical ordering test: vg++ must be expanded before vg+.
        If vg+ is expanded first, vg++ becomes 'very good plus+' (corrupted).
        """
        result = preprocessor.clean_text("VG++ sleeve, VG+ record")
        assert "very good plus+ " not in result
        assert result.count("very good plus") == 2

    def test_ex_expands(self, preprocessor):
        result = preprocessor.clean_text("EX condition")
        assert "excellent" in result

    def test_lowercase_applied(self, preprocessor):
        result = preprocessor.clean_text("Near Mint Sleeve")
        assert result == result.lower()

    def test_whitespace_normalized(self, preprocessor):
        result = preprocessor.clean_text("plays   perfectly")
        assert "  " not in result


class TestStrayNumericTokens:
    def test_strips_boilerplate_lone_digit_before_condition_words(
        self, preprocessor
    ):
        result = preprocessor.clean_text("6 sealed, new hype sticker")
        assert not re.search(r"\b6\b", result)
        assert "sealed" in result

    def test_preserves_two_lp_count(self, preprocessor):
        result = preprocessor.clean_text("2 lp set, plays well")
        assert "2 lp" in result

    def test_preserves_disk_m_of_n(self, preprocessor):
        result = preprocessor.clean_text("disk 2 of 3, light marks")
        assert "2 of" in result

    def test_preserves_fraction(self, preprocessor):
        result = preprocessor.clean_text("plays at 3/4 speed")
        assert "3/4" in result

    def test_preserves_inch_marker(self, preprocessor):
        result = preprocessor.clean_text('7" pressing in nm sleeve')
        assert '7"' in result or "7" in result  # quote normalized by source

    def test_preserves_inch_split_phrase(self, preprocessor):
        result = preprocessor.clean_text('2" split at the spine, vg sleeve')
        assert re.search(r"\b2\b", result)
        assert "split" in result

    def test_preserves_digit_space_quote_inch(self, preprocessor):
        result = preprocessor.clean_text('2 " split seam, nm media')
        assert re.search(r"\b2\b", result)
        assert "split" in result

    def test_preserves_inch_spelled_out(self, preprocessor):
        result = preprocessor.clean_text("6 inch seam split, light ring wear")
        assert re.search(r"\b6\b", result)
        assert "inch" in result
        assert "seam split" in result

    def test_preserves_inches_plural(self, preprocessor):
        result = preprocessor.clean_text("corner ding ~3 inches from edge")
        assert re.search(r"\b3\b", result)
        assert "inches" in result


class TestProtectedTerms:
    def test_protected_terms_built(self, preprocessor):
        assert len(preprocessor.protected_terms) > 0

    def test_sealed_is_protected(self, preprocessor):
        assert "sealed" in preprocessor.protected_terms

    def test_surface_noise_is_protected(self, preprocessor):
        assert "surface noise" in preprocessor.protected_terms

    def test_protected_terms_survive_cleaning(self, preprocessor):
        text   = "sealed, unplayed, no marks"
        cleaned = preprocessor.clean_text(text)
        lost   = preprocessor._verify_protected_terms(text, cleaned)
        assert len(lost) == 0


class TestUnverifiedMediaDetection:
    def test_unplayed_is_unverified(self, preprocessor):
        result = preprocessor.detect_unverified_media("unplayed, still sealed")
        assert result is False

    def test_untested_is_unverified(self, preprocessor):
        result = preprocessor.detect_unverified_media("untested, sold as seen")
        assert result is False

    def test_normal_text_is_verifiable(self, preprocessor):
        result = preprocessor.detect_unverified_media(
            "plays perfectly, light scuff"
        )
        assert result is True

    def test_detection_on_raw_text(self, preprocessor):
        """Detection must work on raw text, before any normalization."""
        result = preprocessor.detect_unverified_media(
            "UNPLAYED, still in shrink"
        )
        assert result is False


class TestGenericDetection:
    def test_generic_sleeve_detected(self, preprocessor):
        result = preprocessor.detect_generic_sleeve(
            "generic white sleeve, die-cut"
        )
        assert result is True

    def test_plain_sleeve_detected(self, preprocessor):
        result = preprocessor.detect_generic_sleeve("plain sleeve only")
        assert result is True

    def test_normal_sleeve_not_detected(self, preprocessor):
        result = preprocessor.detect_generic_sleeve(
            "original cover, near mint condition"
        )
        assert result is False


class TestAdaptiveStratification:
    def test_selects_more_imbalanced_target(
        self, preprocessor, sample_unified_records
    ):
        key = preprocessor.select_stratify_key(sample_unified_records)
        assert key in ["sleeve_label", "media_label"]

    def test_imbalance_ratio_computed(
        self, preprocessor, sample_unified_records
    ):
        preprocessor.select_stratify_key(sample_unified_records)
        assert preprocessor._stats["sleeve_imbalance_ratio"] >= 1.0
        assert preprocessor._stats["media_imbalance_ratio"] >= 1.0

    def test_split_sizes_sum_to_total(
        self, preprocessor, sample_unified_records
    ):
        splits = preprocessor.split_records(sample_unified_records)
        total = sum(len(v) for v in splits.values())
        assert total == len(sample_unified_records)

    def test_split_keys_correct(self, preprocessor, sample_unified_records):
        splits = preprocessor.split_records(sample_unified_records)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_records_tagged_with_split(
        self, preprocessor, sample_unified_records
    ):
        splits = preprocessor.split_records(sample_unified_records)
        for split_name, records in splits.items():
            for record in records:
                assert record["split"] == split_name

    def test_no_record_in_multiple_splits(
        self, preprocessor, sample_unified_records
    ):
        splits = preprocessor.split_records(sample_unified_records)
        all_ids = []
        for records in splits.values():
            all_ids.extend([r["item_id"] for r in records])
        assert len(all_ids) == len(set(all_ids))


class TestProcessRecord:
    def test_text_clean_added(self, preprocessor, sample_unified_records):
        result = preprocessor.process_record(sample_unified_records[0])
        assert "text_clean" in result

    def test_original_text_preserved(
        self, preprocessor, sample_unified_records
    ):
        record = sample_unified_records[0]
        result = preprocessor.process_record(record)
        assert result["text"] == record["text"]

    def test_labels_unchanged(self, preprocessor, sample_unified_records):
        record = sample_unified_records[0]
        result = preprocessor.process_record(record)
        assert result["sleeve_label"] == record["sleeve_label"]
        assert result["media_label"]  == record["media_label"]

    def test_media_verifiable_updated(
        self, preprocessor, sample_unified_records
    ):
        record = {
            **sample_unified_records[0],
            "text": "untested, sold as is",
        }
        result = preprocessor.process_record(record)
        assert result["media_verifiable"] is False
