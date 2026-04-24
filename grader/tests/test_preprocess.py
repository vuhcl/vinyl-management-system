"""
grader/tests/test_preprocess.py

Tests for preprocess.py — text normalization, abbreviation expansion,
protected term preservation, unverified media detection, and
adaptive stratified splitting.
"""

import re
from pathlib import Path

import pytest
import yaml

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
        text = "sealed, unplayed, no marks"
        cleaned = preprocessor.clean_text(text)
        lost = preprocessor._verify_protected_terms(text, cleaned)
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

    def test_media_unmentioned_is_unverified(self, preprocessor):
        # Sleeve-only language: describes cover defects, no playback cues.
        result = preprocessor.detect_unverified_media(
            "seam split on cover; small corner crease"
        )
        assert result is False

    def test_sealed_is_exempt_and_verified(self, preprocessor):
        # In this project, sealed implies Mint media by convention.
        result = preprocessor.detect_unverified_media(
            "factory sealed, no play info provided"
        )
        assert result is True

    def test_detection_on_raw_text(self, preprocessor):
        """Detection must work on raw text, before any normalization."""
        result = preprocessor.detect_unverified_media(
            "UNPLAYED, still in shrink"
        )
        assert result is False

    def test_mixed_comment_vinyl_surface_marks_is_verifiable(self, preprocessor):
        result = preprocessor.detect_unverified_media(
            "Vinyl has some light surface marks, a few pressing dimples, "
            "small drill hole in center label."
        )
        assert result is True

    def test_mixed_comment_minor_play_wear_is_verifiable(self, preprocessor):
        result = preprocessor.detect_unverified_media(
            "Gently used copy in nice condition. Some very minor play wear. "
            "Labels have some very minor bubbling likely present at press"
        )
        assert result is True

    def test_vague_comment_nice_overall_is_unverified(self, preprocessor):
        result = preprocessor.detect_unverified_media("Nice overall")
        assert result is False

    def test_vague_comment_great_shape_all_around_is_unverified(
        self, preprocessor
    ):
        result = preprocessor.detect_unverified_media("Great shape all around")
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
        assert result["media_label"] == record["media_label"]

    def test_media_verifiable_updated(
        self, preprocessor, sample_unified_records
    ):
        record = {
            **sample_unified_records[0],
            "text": "untested, sold as is",
        }
        result = preprocessor.process_record(record)
        assert result["media_verifiable"] is False


class TestDescriptionQuality:
    def test_rich_note_adequate_for_training(self, preprocessor):
        text = (
            "Corner bump and light ring wear on cover; "
            "vinyl plays cleanly with faint surface noise."
        )
        cleaned = preprocessor.clean_text(text)
        dq = preprocessor.compute_description_quality(text, cleaned)
        assert dq["sleeve_note_adequate"] is True
        assert dq["media_note_adequate"] is True
        assert dq["adequate_for_training"] is True
        assert dq["needs_richer_note"] is False

    def test_sleeve_only_thin_note(self, preprocessor):
        text = (
            "Light seam split on the jacket. "
            "Seller gave no playback or vinyl condition details."
        )
        cleaned = preprocessor.clean_text(text)
        dq = preprocessor.compute_description_quality(text, cleaned)
        assert dq["sleeve_note_adequate"] is True
        assert dq["media_note_adequate"] is False
        assert dq["adequate_for_training"] is False
        assert "media" in dq["description_quality_gaps"]

    def test_grade_shorthand_sleeve_ok_media_thin(self, preprocessor):
        text = "NM / VG+"
        cleaned = preprocessor.clean_text(text)
        dq = preprocessor.compute_description_quality(text, cleaned)
        assert dq["sleeve_note_adequate"] is True
        assert dq["media_note_adequate"] is False

    def test_mint_mint_short_note_sleeve_relaxed_when_enabled(
        self, test_config, guidelines_path
    ):
        with Path(test_config).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["preprocessing"]["description_adequacy"][
            "mint_sleeve_label_relax_sleeve_note"
        ] = True
        pre = Preprocessor("unused.yaml", guidelines_path, config=cfg)
        text = "brand new, sealed"
        cleaned = pre.clean_text(text)
        dq = pre.compute_description_quality(
            text, cleaned, sleeve_label="Mint", media_label="Mint"
        )
        assert dq["sleeve_note_adequate"] is True
        assert dq["media_note_adequate"] is True
        assert dq["adequate_for_training"] is True

    def test_mint_sleeve_near_mint_media_short_note_relaxed_when_enabled(
        self, test_config, guidelines_path
    ):
        with Path(test_config).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["preprocessing"]["description_adequacy"][
            "mint_sleeve_label_relax_sleeve_note"
        ] = True
        pre = Preprocessor("unused.yaml", guidelines_path, config=cfg)
        text = "brand new, sealed"
        cleaned = pre.clean_text(text)
        dq = pre.compute_description_quality(
            text,
            cleaned,
            sleeve_label="Mint",
            media_label="Near Mint",
        )
        assert dq["sleeve_note_adequate"] is True
        assert dq["media_note_adequate"] is True
        assert dq["adequate_for_training"] is True

    def test_mint_mint_relax_off_short_note_still_thin_sleeve(
        self, preprocessor
    ):
        text = "brand new, sealed"
        cleaned = preprocessor.clean_text(text)
        dq = preprocessor.compute_description_quality(
            text, cleaned, sleeve_label="Mint", media_label="Mint"
        )
        assert dq["sleeve_note_adequate"] is False

    def test_near_mint_sleeve_short_note_not_relaxed_even_when_enabled(
        self, test_config, guidelines_path
    ):
        with Path(test_config).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["preprocessing"]["description_adequacy"][
            "mint_sleeve_label_relax_sleeve_note"
        ] = True
        pre = Preprocessor("unused.yaml", guidelines_path, config=cfg)
        text = "brand new, sealed"
        cleaned = pre.clean_text(text)
        dq = pre.compute_description_quality(
            text,
            cleaned,
            sleeve_label="Near Mint",
            media_label="Mint",
        )
        assert dq["sleeve_note_adequate"] is False

    def test_legacy_mint_both_labels_config_key_still_honored(
        self, test_config, guidelines_path
    ):
        with Path(test_config).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        da = cfg["preprocessing"]["description_adequacy"]
        da.pop("mint_sleeve_label_relax_sleeve_note", None)
        da["mint_both_labels_relax_sleeve_note"] = True
        pre = Preprocessor("unused.yaml", guidelines_path, config=cfg)
        text = "brand new, sealed"
        cleaned = pre.clean_text(text)
        dq = pre.compute_description_quality(
            text,
            cleaned,
            sleeve_label="Mint",
            media_label="Very Good Plus",
        )
        assert dq["sleeve_note_adequate"] is True

    def test_process_record_has_quality_fields(
        self, preprocessor, sample_unified_records
    ):
        r = preprocessor.process_record(sample_unified_records[0])
        assert "sleeve_note_adequate" in r
        assert "description_quality_prompts" in r

    def test_process_record_mint_sleeve_short_note_when_relax_on(
        self, test_config, guidelines_path
    ):
        with Path(test_config).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["preprocessing"]["description_adequacy"][
            "mint_sleeve_label_relax_sleeve_note"
        ] = True
        pre = Preprocessor("unused.yaml", guidelines_path, config=cfg)
        r = pre.process_record(
            {
                "item_id": "mint_short",
                "source": "discogs",
                "text": "brand new, sealed",
                "sleeve_label": "Mint",
                "media_label": "Near Mint",
            }
        )
        assert r["adequate_for_training"] is True


@pytest.mark.usefixtures("unified_jsonl_path")
class TestClassDistributionSplitsReport:
    def test_writes_class_distribution_splits_report(
        self, test_config, guidelines_path, tmp_dirs
    ):
        preprocessor = Preprocessor(test_config, guidelines_path)
        preprocessor.run()
        path = tmp_dirs["reports"] / "class_distribution_splits.txt"
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "CLASS DISTRIBUTION BY SPLIT (AFTER PREPROCESS)" in text
        assert "Full pool (all rows written to preprocessed.jsonl)" in text
        assert "Split: train" in text
        assert "Split: val" in text
        assert "Split: test" in text
        assert "Grade" in text and "Sleeve" in text and "Media" in text
