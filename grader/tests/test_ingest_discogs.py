"""
grader/tests/test_ingest_discogs.py

Tests for ingest_discogs.py — grade normalization, drop logic,
unverified media detection, and output schema.
"""

import pytest

from grader.src.data.ingest_discogs import (
    DiscogsIngester,
    _DEFAULT_GENERIC_NOTE_PATTERNS,
    _DEFAULT_ITEM_SPECIFIC_HINTS,
    _DEFAULT_PRESERVATION_KEYWORDS,
    normalize_seller_comment_text,
)


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

    def test_generic_normalizes(self, ingester):
        result = ingester.normalize_grade("Generic")
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


class TestGenericSellerNotes:
    """generic_note_filter behavior (enabled on a dedicated ingester)."""

    @pytest.fixture
    def ingester_generic(self, ingester):
        ingester.generic_note_filter_enabled = True
        ingester.generic_note_patterns = [
            "all records are visually graded under bright light",
            "please refer to discogs grading",
        ]
        ingester.item_specific_hints = ["scuff", "scratch"]
        ingester.preservation_keywords = ["sealed", "shrink", "brand new"]
        return ingester

    def test_drops_boilerplate_only(self, ingester_generic):
        reason = ingester_generic._get_drop_reason(
            "All records are visually graded under bright light. "
            "Please see our other listings!",
            "Near Mint (NM or M-)",
            "Very Good Plus (VG+)",
        )
        assert reason == "generic_seller_notes"

    def test_keeps_when_item_specific_hint_present(self, ingester_generic):
        reason = ingester_generic._get_drop_reason(
            "Visually graded under bright light; light scuff on side A only.",
            "Near Mint (NM or M-)",
            "Very Good Plus (VG+)",
        )
        assert reason is None

    def test_keeps_when_media_mint(self, ingester_generic):
        reason = ingester_generic._get_drop_reason(
            "All records are visually graded. Ships within 2 days.",
            "Near Mint (NM or M-)",
            "Mint (M)",
        )
        assert reason is None

    def test_keeps_when_sleeve_mint(self, ingester_generic):
        reason = ingester_generic._get_drop_reason(
            "All records are visually graded.",
            "Mint (M)",
            "Very Good Plus (VG+)",
        )
        assert reason is None

    def test_keeps_when_sealed_in_comment(self, ingester_generic):
        reason = ingester_generic._get_drop_reason(
            "Still factory sealed. All records are visually graded.",  # pattern + sealed
            "Mint (M)",
            "Mint (M)",
        )
        assert reason is None

    def test_keeps_when_brand_new_in_comment(self, ingester_generic):
        reason = ingester_generic._get_drop_reason(
            "Brand new copy. Please refer to discogs grading standards.",
            "Near Mint (NM or M-)",
            "Near Mint (NM or M-)",
        )
        assert reason is None


class TestNormalizeSellerCommentText:
    def test_strips_urls_www_and_links(self):
        s = "links: https://example.com/a and www.test.org/x ok"
        out = normalize_seller_comment_text(s)
        assert "http" not in out.lower()
        assert "www." not in out.lower()
        assert "links" not in out.lower()
        assert "ok" in out.lower()

    def test_strips_emoji_marks_underscores(self):
        s = "________vg+ with ★ and 💯 and hairline"
        out = normalize_seller_comment_text(s)
        assert "💯" not in out
        assert "★" not in out
        assert "___" not in out
        assert "hairline" in out.lower()
        assert "vg+" in out.lower()

    def test_respects_strip_url_off(self):
        t = normalize_seller_comment_text(
            "a https://x.com b", strip_urls=False, strip_emoji=True
        )
        assert "https" in t

    def test_respects_strip_emoji_off(self):
        t = normalize_seller_comment_text(
            "a ★ b", strip_urls=True, strip_emoji=False
        )
        assert "★" in t

    def test_disables_both_strips(self):
        raw = "see https://a.com/ ★"
        t = normalize_seller_comment_text(raw, strip_urls=False, strip_emoji=False)
        assert "https" in t
        assert "★" in t


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


class TestStripBoilerplateNotes:
    @pytest.fixture
    def ingester_strip(self, ingester):
        ingester.strip_boilerplate_enabled = True
        ingester.generic_note_patterns = list(_DEFAULT_GENERIC_NOTE_PATTERNS)
        ingester.item_specific_hints = list(_DEFAULT_ITEM_SPECIFIC_HINTS)
        ingester.preservation_keywords = list(_DEFAULT_PRESERVATION_KEYWORDS)
        return ingester

    def test_removes_shipping_sentence_keeps_defects(self, ingester_strip):
        raw = (
            "Light hairline on side A under bright light. "
            "We ship within 2 business days."
        )
        out = ingester_strip.strip_boilerplate_from_notes(raw)
        assert "hairline" in out.lower()
        assert "ship" not in out.lower()

    def test_preservation_sentence_kept_even_with_grading_pattern(
        self, ingester_strip
    ):
        raw = (
            "Still in shrink. All records are visually graded under bright light."
        )
        out = ingester_strip.strip_boilerplate_from_notes(raw)
        assert "shrink" in out.lower()
        assert "visually graded" not in out.lower()

    def test_secondhand_profile_policies_then_real_defect(self, ingester_strip):
        """Real Discogs-style template: boilerplate sentences + condition line."""
        raw = (
            "Secondhand item in our store. Please visit our profile for general "
            "information including shipping prices and policies. "
            "bottom right corner dinged."
        )
        out = ingester_strip.strip_boilerplate_from_notes(raw)
        assert "dinged" in out.lower()
        assert "corner" in out.lower()
        assert "secondhand" not in out.lower()
        assert "visit our profile" not in out.lower()
        assert "shipping prices" not in out.lower()

    def test_mixed_segment_strips_policy_substring(self, ingester_strip):
        raw = (
            "With Obi and 2-Inserts(mold stains) and Inner Sleeve. "
            "Accessories (obi, liner notes, etc.) are mentioned in the item "
            "description only if included. For items over $20, contact us for "
            "details and pictures."
        )
        out = ingester_strip.strip_boilerplate_from_notes(raw).lower()
        assert "mold stains" in out
        assert "with obi" in out
        assert "for items over" not in out
        assert "details and pictures" not in out
        assert "mentioned in the item description" not in out

    def test_strips_ships_same_day_boilerplate(self, ingester_strip):
        raw = (
            "Light hairline on side a. We offer ships same day when paid before noon."
        )
        out = ingester_strip.strip_boilerplate_from_notes(raw).lower()
        assert "hairline" in out
        assert "ships same day" not in out

    def test_strips_buy_five_get_cheapest_free(self, ingester_strip):
        raw = (
            "Light hairline on side a. * buy 5 get cheapest free * "
            "Extra text about nothing important."
        )
        out = ingester_strip.strip_boilerplate_from_notes(raw).lower()
        assert "hairline" in out
        assert "buy 5" not in out
        assert "get cheapest" not in out

    def test_strips_promo_and_refund_spam_keeps_condition(self, ingester_strip):
        raw = (
            "[HALF OFF! Marked for deletion 3/23] [NEW LOW PRICE MARCH 15] "
            "[From Pittsburgh George's Collection] FROM A REPUTABLE SELLER! "
            "$6 / additional LP. "
            "+++++FULL REFUNDS AVAILABLE IF UNHAPPY++BUY WITH CONFIDENCE "
            "Light corner wear, plays cleanly."
        )
        out = ingester_strip.strip_boilerplate_from_notes(raw).lower()
        assert "corner wear" in out or "plays cleanly" in out
        assert "half off" not in out
        assert "marked for deletion" not in out
        assert "new low price" not in out
        assert "reputable seller" not in out
        assert "buy with confidence" not in out
        assert "refunds available" not in out
        assert "collection]" not in out
        assert "$6" not in out

    def test_parse_listing_uses_stripped_text(self, ingester_strip, monkeypatch):
        monkeypatch.setattr(
            ingester_strip,
            "strip_boilerplate_from_notes",
            lambda t: "plays perfectly, light scuff only",
        )
        listing = {
            "id": 1,
            "condition": "Near Mint (NM or M-)",
            "sleeve_condition": "Very Good Plus (VG+)",
            "comments": "IGNORED BY_PATCH",
            "release": {"artist": "A", "title": "T"},
        }
        ingester_strip._stats = {"drops": {}}
        r = ingester_strip.parse_listing(listing)
        assert r is not None
        assert r["text"] == "plays perfectly, light scuff only"


class TestParseListing:
    def test_parse_listing_strips_condition_whitespace(self, ingester):
        """Padded Discogs / SQLite condition strings still map via condition_map."""
        ingester._stats = {"drops": {}}
        listing = {
            "id": 999,
            "condition": "  Near Mint (NM or M-)  ",
            "sleeve_condition": " Very Good Plus (VG+) ",
            "comments": (
                "Light hairline under bright light; jacket shows minor ring wear "
                "and plays cleanly with no pops or skips."
            ),
            "release": {"artist": "A", "title": "T"},
        }
        r = ingester.parse_listing(listing)
        assert r is not None
        assert r["sleeve_label"] == "Very Good Plus"
        assert r["media_label"] == "Near Mint"
        assert r["raw_sleeve"] == "Very Good Plus (VG+)"
        assert r["raw_media"] == "Near Mint (NM or M-)"

    def test_parse_listing_strips_urls_before_text(self, ingester):
        """``normalize_seller_comment_text`` runs before ``strip_boilerplate``."""
        ingester._stats = {"drops": {}}
        listing = {
            "id": 1,
            "condition": "Near Mint (NM or M-)",
            "sleeve_condition": "Very Good Plus (VG+)",
            "comments": (
                "Light hairline under bright light and see photos at "
                "https://example.com/photo for detail."
            ),
            "release": {"artist": "A", "title": "T"},
        }
        r = ingester.parse_listing(listing)
        assert r is not None
        assert "http" not in r["text"]
        assert "https" not in r["text"]
        assert "hairline" in r["text"].lower()

    def test_valid_listing_parsed(self, ingester, sample_discogs_listing):
        ingester._stats = {"drops": {}}
        result = ingester.parse_listing(sample_discogs_listing)
        assert result is not None
        assert result["source"] == "discogs"
        assert result["sleeve_label"] == "Very Good Plus"
        assert result["media_label"] == "Near Mint"

    def test_output_schema_complete(self, ingester, sample_discogs_listing):
        ingester._stats = {"drops": {}}
        result = ingester.parse_listing(sample_discogs_listing)
        required_fields = [
            "item_id", "source", "text", "sleeve_label", "media_label",
            "label_confidence", "media_verifiable", "obi_condition",
            "raw_sleeve", "raw_media", "artist", "title",
            "release_format", "release_description",
        ]
        for field in required_fields:
            assert field in result

    def test_parse_listing_includes_release_format_from_api(self, ingester):
        ingester._stats = {"drops": {}}
        listing = {
            "id": 99,
            "condition": "Near Mint (NM or M-)",
            "sleeve_condition": "Very Good Plus (VG+)",
            "comments": "plays perfectly, no issues",
            "release": {
                "artist": "A",
                "title": "T",
                "format": "Vinyl, LP, Album",
                "description": "180g reissue",
            },
        }
        r = ingester.parse_listing(listing)
        assert r is not None
        assert r["release_format"] == "Vinyl, LP, Album"
        assert r["release_description"] == "180g reissue"

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
        monkeypatch.setattr(
            "grader.src.data.ingest_discogs.load_project_dotenv",
            lambda: None,
        )
        with pytest.raises(EnvironmentError):
            DiscogsIngester(
                config_path=test_config,
                guidelines_path=guidelines_path,
            )

    def test_fetch_all_requires_inventory_sellers(self, ingester):
        ingester.inventory_sellers = []
        with pytest.raises(ValueError, match="inventory_sellers"):
            ingester.fetch_all()


class TestPublicInventoryPageCap:
    def test_skips_fetch_above_cap(self, ingester, monkeypatch):
        called: list[int] = []

        def boom(*_a, **_k):
            called.append(1)
            raise AssertionError("_get should not run above page cap")

        monkeypatch.setattr(ingester, "_get", boom)
        ingester.max_public_inventory_pages = 3
        out = ingester.fetch_inventory_page("anyone", page=4)
        assert out["listings"] == []
        assert called == []


class TestCacheOnlyMode:
    def test_does_not_call_network_when_missing_page(self, ingester, monkeypatch):
        # Force cache-only mode; tmp dirs are empty so page_001 is missing.
        ingester.cache_only = True

        def boom(*_a, **_k):
            raise AssertionError("_get should not be called in cache-only mode")

        monkeypatch.setattr(ingester, "_get", boom)
        out = ingester.fetch_inventory_page("anyone", page=1)
        assert out["listings"] == []


class TestFormatFilter:
    def test_vinyl_release_matches(self, ingester):
        assert ingester._listing_matches_format_filter(
            {"release": {"format": "(LP, Album)", "description": "Artist - Title"}}
        )

    def test_cd_release_does_not_match_vinyl_filter(self, ingester):
        assert not ingester._listing_matches_format_filter(
            {"release": {"format": "(CD, Album)", "description": "Artist - Title"}}
        )


class TestOfflineParseOnly:
    def test_no_discogs_token_required(
        self, test_config, guidelines_path, monkeypatch
    ):
        monkeypatch.delenv("DISCOGS_TOKEN", raising=False)
        g = DiscogsIngester(
            test_config,
            guidelines_path,
            offline_parse_only=True,
        )
        assert g.session is None
        assert g.offline_parse_only is True

    def test_parse_release_marketplace_listing_dict(
        self, test_config, guidelines_path, monkeypatch
    ):
        monkeypatch.delenv("DISCOGS_TOKEN", raising=False)
        g = DiscogsIngester(
            test_config,
            guidelines_path,
            offline_parse_only=True,
        )
        listing = {
            "id": 888001,
            "sleeve_condition": "Very Good (VG)",
            "condition": "Near Mint (NM or M-)",
            "comments": (
                "Light ring wear on the cover corners; vinyl plays cleanly "
                "with only faint surface noise on the run-in groove."
            ),
            "release": {
                "artist": "Test Artist",
                "title": "Test Title",
                "year": 1975,
                "country": "US",
                "format": "LP",
                "description": "",
            },
        }
        row = g.parse_listing(listing)
        assert row is not None
        assert row["item_id"] == "888001"
        assert row["sleeve_label"] == "Very Good"
        assert row["media_label"] == "Near Mint"
