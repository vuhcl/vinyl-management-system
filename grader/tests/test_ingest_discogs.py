"""
grader/tests/test_ingest_discogs.py

Tests for ingest_discogs.py — grade normalization, drop logic,
unverified media detection, and output schema.
"""

import os

import pytest

from grader.src.data.ingest_discogs import DiscogsIngester


@pytest.fixture
def ingester(test_config, guidelines_path, monkeypatch):
    monkeypatch.setenv("DISCOGS_TOKEN", "TEST_TOKEN")
    return DiscogsIngester(
        config_path=test_config,
        guidelines_path=guidelines_path,
    )


class TestGradeNormalization:
    def test_near_mint_normalizes(self, ingester):
        result = ingester.normalize_grade("Near Mint (NM or M-)")
        assert result == "Near Mint"

    def test_very_good_plus_normalizes(self, ingester):
        result = ingester.normalize_grade("Very Good Plus (VG+)")
        assert result == "Very Good Plus"

    def test_good_plus_normalizes_to_good(self, ingester):
        result = ingester.normalize_grade("Good Plus (G+)")
        assert result == "Good"

    def test_good_normalizes_to_good(self, ingester):
        result = ingester.normalize_grade("Good (G)")
        assert result == "Good"

    def test_fair_normalizes_to_poor(self, ingester):
        result = ingester.normalize_grade("Fair (F)")
        assert result == "Poor"

    def test_poor_normalizes_to_poor(self, ingester):
        result = ingester.normalize_grade("Poor (P)")
        assert result == "Poor"

    def test_generic_sleeve_normalizes(self, ingester):
        result = ingester.normalize_grade("Generic Sleeve")
        assert result == "Generic"

    def test_unknown_condition_returns_none(self, ingester):
        result = ingester.normalize_grade("Unknown Grade (UG)")
        assert result is None


class TestDropLogic:
    def test_empty_notes_dropped(self, ingester):
        reason = ingester._get_drop_reason("", "Near Mint (NM or M-)", "Very Good Plus (VG+)")
        assert reason == "missing_notes"

    def test_short_notes_dropped(self, ingester):
        reason = ingester._get_drop_reason("ok", "Near Mint (NM or M-)", "Very Good Plus (VG+)")
        assert reason == "notes_too_short"

    def test_missing_sleeve_dropped(self, ingester):
        reason = ingester._get_drop_reason("plays perfectly", "", "Very Good Plus (VG+)")
        assert reason == "missing_sleeve_condition"

    def test_missing_media_dropped(self, ingester):
        reason = ingester._get_drop_reason("plays perfectly", "Very Good Plus (VG+)", "")
        assert reason == "missing_media_condition"

    def test_valid_listing_not_dropped(self, ingester):
        reason = ingester._get_drop_reason(
            "plays perfectly, light scuff only",
            "Near Mint (NM or M-)",
            "Very Good Plus (VG+)",
        )
        assert reason is None


class TestUnverifiedMediaDetection:
    def test_unplayed_is_unverified(self, ingester):
        result = ingester._detect_media_verifiable("sealed, unplayed")
        assert result is False

    def test_untested_is_unverified(self, ingester):
        result = ingester._detect_media_verifiable("untested, sold as seen")
        assert result is False

    def test_normal_notes_are_verifiable(self, ingester):
        result = ingester._detect_media_verifiable("plays perfectly, VG+")
        assert result is True


class TestParseListing:
    def test_valid_listing_parsed(self, ingester, sample_discogs_listing):
        ingester._stats = {"drops": {}}
        result = ingester.parse_listing(sample_discogs_listing)
        assert result is not None
        assert result["source"] == "discogs"
        assert result["sleeve_label"] == "Very Good Plus"
        assert result["media_label"]  == "Near Mint"

    def test_output_schema_complete(self, ingester, sample_discogs_listing):
        ingester._stats = {"drops": {}}
        result = ingester.parse_listing(sample_discogs_listing)
        required_fields = [
            "item_id", "source", "text", "sleeve_label", "media_label",
            "label_confidence", "media_verifiable", "obi_condition",
            "raw_sleeve", "raw_media", "artist", "title",
        ]
        for field in required_fields:
            assert field in result

    def test_label_confidence_is_one(self, ingester, sample_discogs_listing):
        ingester._stats = {"drops": {}}
        result = ingester.parse_listing(sample_discogs_listing)
        assert result["label_confidence"] == 1.0

    def test_obi_condition_is_none(self, ingester, sample_discogs_listing):
        ingester._stats = {"drops": {}}
        result = ingester.parse_listing(sample_discogs_listing)
        assert result["obi_condition"] is None


class TestInitialization:
    def test_missing_token_raises(self, test_config, guidelines_path, monkeypatch):
        monkeypatch.delenv("DISCOGS_TOKEN", raising=False)
        with pytest.raises(EnvironmentError):
            DiscogsIngester(
                config_path=test_config,
                guidelines_path=guidelines_path,
            )
