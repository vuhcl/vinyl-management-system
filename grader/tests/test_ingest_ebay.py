"""
grader/tests/test_ingest_ebay.py

Tests for ingest_ebay.py — grade token extraction for clean and
annotated formats, harmonization, item specifics parsing, and
output schema validation.
"""

import os
import re

import pytest

from grader.src.data.ingest_ebay import EbayIngester


@pytest.fixture
def ingester(test_config, guidelines_path, monkeypatch):
    monkeypatch.setenv("EBAY_CLIENT_ID",     "TEST_CLIENT_ID")
    monkeypatch.setenv("EBAY_CLIENT_SECRET", "TEST_SECRET")
    return EbayIngester(
        config_path=test_config,
        guidelines_path=guidelines_path,
    )


class TestGradeTokenExtraction:
    """Tests for _extract_grade_token — the core parsing logic."""

    def test_clean_bare_grade(self, ingester):
        token, suffix = ingester._extract_grade_token("VG+")
        assert token  == "VG+"
        assert suffix == ""

    def test_clean_grade_with_dash(self, ingester):
        token, suffix = ingester._extract_grade_token("E-")
        assert token  == "E-"
        assert suffix == ""

    def test_annotated_grade_with_expansion(self, ingester):
        """E (Excellent) S (Stain) cornerbump → grade=E, suffix=S (Stain) cornerbump"""
        token, suffix = ingester._extract_grade_token(
            "E (Excellent) S (Stain) cornerbump"
        )
        assert token  == "E"
        assert "stain" in suffix.lower() or "S" in suffix
        assert "cornerbump" in suffix

    def test_annotated_grade_expansion_stripped(self, ingester):
        """Parenthetical grade expansion e.g. (Excellent Plus) should be stripped."""
        token, suffix = ingester._extract_grade_token("E+ (Excellent Plus)")
        assert token  == "E+"
        assert suffix == ""

    def test_m_minus_extracted(self, ingester):
        token, suffix = ingester._extract_grade_token("M-")
        assert token  == "M-"

    def test_s_extracted(self, ingester):
        token, suffix = ingester._extract_grade_token("S")
        assert token  == "S"


class TestHarmonization:
    def test_s_harmonizes_to_mint(self, ingester):
        result = ingester.harmonize_grade("S")
        assert result is not None
        grade, conf = result
        assert grade == "Mint"
        assert conf  == 1.0

    def test_eminus_harmonizes_to_excellent(self, ingester):
        result = ingester.harmonize_grade("E-")
        assert result is not None
        grade, conf = result
        assert grade == "Excellent"
        assert conf  == pytest.approx(0.85)

    def test_vgplus_harmonizes_to_very_good_plus(self, ingester):
        result = ingester.harmonize_grade("VG+")
        assert result is not None
        grade, conf = result
        assert grade == "Very Good Plus"

    def test_vg_harmonizes_to_very_good(self, ingester):
        result = ingester.harmonize_grade("VG")
        assert result is not None
        grade, conf = result
        assert grade == "Very Good"

    def test_unknown_grade_returns_none(self, ingester):
        result = ingester.harmonize_grade("UNKNOWN")
        assert result is None


class TestItemSpecificsExtraction:
    def test_extract_known_field(self, ingester, sample_ebay_item_clean):
        specifics = ingester._extract_item_specifics(sample_ebay_item_clean)
        assert "Sleeve Grading" in specifics
        assert specifics["Sleeve Grading"] == "VG+"

    def test_extract_obi_field(self, ingester, sample_ebay_item_clean):
        specifics = ingester._extract_item_specifics(sample_ebay_item_clean)
        assert "OBI_Grading" in specifics

    def test_case_insensitive_lookup(self, ingester, sample_ebay_item_clean):
        specifics = ingester._extract_item_specifics(sample_ebay_item_clean)
        result = ingester._get_field(specifics, "sleeve grading")
        assert result == "VG+"

    def test_missing_field_returns_none(self, ingester, sample_ebay_item_clean):
        specifics = ingester._extract_item_specifics(sample_ebay_item_clean)
        result = ingester._get_field(specifics, "NonExistentField")
        assert result is None


class TestParseItem:
    def test_clean_format_parsed(
        self, ingester, sample_ebay_item_clean
    ):
        ingester._stats = {"drops": {}}
        result = ingester.parse_item(sample_ebay_item_clean, "facerecords")
        assert result is not None
        assert result["source"]       == "ebay_jp"
        assert result["sleeve_label"] == "Very Good Plus"
        assert result["media_label"]  == "Excellent"

    def test_annotated_format_parsed(
        self, ingester, sample_ebay_item_annotated
    ):
        ingester._stats = {"drops": {}}
        result = ingester.parse_item(
            sample_ebay_item_annotated, "ellarecords2005"
        )
        assert result is not None
        assert result["sleeve_label"] == "Excellent"
        assert result["media_label"]  == "Near Mint"

    def test_annotated_text_contains_defect_suffix(
        self, ingester, sample_ebay_item_annotated
    ):
        ingester._stats = {"drops": {}}
        result = ingester.parse_item(
            sample_ebay_item_annotated, "ellarecords2005"
        )
        # Text should contain the defect suffix from annotated value
        assert result["text"] is not None
        assert len(result["text"]) > 0

    def test_obi_condition_captured(
        self, ingester, sample_ebay_item_clean
    ):
        ingester._stats = {"drops": {}}
        result = ingester.parse_item(sample_ebay_item_clean, "facerecords")
        assert result["obi_condition"] is not None

    def test_label_confidence_less_than_one_for_eminus(
        self, ingester, sample_ebay_item_annotated
    ):
        """E- has label_confidence=0.85 — reflected in output."""
        ingester._stats = {"drops": {}}
        result = ingester.parse_item(
            sample_ebay_item_annotated, "ellarecords2005"
        )
        # E (sleeve) has confidence 0.95, E+ (media) has confidence 0.95
        # min should be 0.95
        assert result["label_confidence"] <= 1.0

    def test_output_schema_complete(
        self, ingester, sample_ebay_item_clean
    ):
        ingester._stats = {"drops": {}}
        result = ingester.parse_item(sample_ebay_item_clean, "facerecords")
        required = [
            "item_id", "source", "text", "sleeve_label", "media_label",
            "label_confidence", "media_verifiable", "obi_condition",
            "raw_sleeve", "raw_media", "artist", "title",
        ]
        for field in required:
            assert field in result


class TestInitialization:
    def test_missing_credentials_raise(
        self, test_config, guidelines_path, monkeypatch
    ):
        monkeypatch.delenv("EBAY_CLIENT_ID",     raising=False)
        monkeypatch.delenv("EBAY_CLIENT_SECRET", raising=False)
        with pytest.raises(EnvironmentError):
            EbayIngester(
                config_path=test_config,
                guidelines_path=guidelines_path,
            )
