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

from grader.src.data.preprocess import (
    Preprocessor,
    load_promo_noise_patterns,
    strip_listing_promo_noise,
)


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

    def test_abbrev_space_before_following_word_when_input_glued(self, preprocessor):
        """``vg+original`` must not become ``… plusoriginal`` (missing space)."""
        out = preprocessor.clean_text("SOLID VG+Original sleeve").strip()
        assert "very good plus original" in out
        assert "plusoriginal" not in out

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


class TestPromoAndShippingNoiseStripping:
    """Regression corpus for listing promo / shipping boilerplate (plan)."""

    CORPUS = [
        "[february frenzy - up to 90% off! original price: 1]",
        " / $6.40 unlimited us-shipping / free on $100 orders of 3+ items read "
        "seller terms before paying",
        "everything has been marked down 75% for another 24 hours ship up to 20 "
        "records in usa for only $5!! all orders over $25 cleaned on vpi!",
        "summer sale! all vinyl marked down 20%+ unlimited $5--",
        "summer sale 10% off storewide 1 week only!",
        "**all $1 & $2 items = buy 2 get 1 free !! note: price deduction will be "
        "made on the least priced items in the order post-invoice so please "
        "refrain from making payment until final subtotal is adjusted**",
        "***free shipping to uk mainland on orders over £15.00 & on orders over "
        "£50.00 to europe***",
        "### price now reduced to make way for new stock ### ",
        "[sale! 4for3 on everything! sale! 22,000+ items]",
        "part of my personal collection",
        "/ $5.90 unlimited us-shipping / free on $100 orders of 3+ items / read "
        "seller terms before paying",
        " / $7.40 unlimited us-shipping / free on $100 orders of 3+ items read "
        "seller terms before paying",
        "$7.50 shipping for unlimited items in usa! packed safely, shipped "
        "promptly! lp's are shipped in custom boxes for reinforced protection",
        "**all items sent securely in a double padded mailer with the vinyl "
        "separated from the sleeve (unless sealed)**",
        " / $5 unlimited us-shipping / free on $100 orders of 3+ items read "
        "seller terms before paying",
        "customs friendly, all the products that we sell are 100% guaranteed if "
        "not completely satisfied send back for a full refund at our expense we "
        "have a warehouse full of new cd's, cassettes, lp's, 45's, 12'' singles "
        "that are 35 plus years old",
        "warehouse back stock",
        "always shipped with domestic tracking",
        "[1 euro sale: 250,000+ records at 1.00 free shipping on orders above 100 "
        "euro inside eu!]",
        "| pick up order over £10 (cash only) welcome at our shop in hackney "
        "wick, east london",
        "with you within 6",
        "jacksonville pressing",
        "all fair offers accepted",
        "⭐",
        "$5 unlimited shipping in usa",
        "- 1000's more records & cds at our shop in upminster essex",
        "disc stored in anti static inner",
        "ultrasonic cleaned",
        "vpi vacuumed",
        "shipped in sturdy whiplash mailer",
        "whiplash mailer",
        '/all sealed items are sold "one way / as is" and cannot be returned or '
        "exchanged",
        "all orders in uk sent first class",
        "recorded delivery",
        "europe/worldwide with tracking",
        "we only use quality mailers",
        "ship throughout the week",
        "we do this professionally",
        "postal charges reflect quality care and services used",
        "we're marrs plectrum records official rsd real world indie shop in "
        "peterborough uk",
        "pics available upon request",
        "cheapest price",
        "accepting paypal credit",
        "pay in 3",
        "watch my cleaning process here",
        "orders over $60 ship for $6",
        "free shipping on usa orders over $30",
        "flatrate shipping rates to all 6 continents",
        "from our us-hub",
        "check my other black sabbath records and combine shipping !!!",
        "buy 12 records and get the cheapest for free",
        "you will receive 2 free records in the same style",
        "free shipping: above 145 euro in europe (eu)",
        "check out our big stock of house techno trance disco & more",
        "pick up in barcelona possible",
        "cheap worldwide shipping price",
        "regular 1-5lp is the same shipping cost (2lp count as 2) gatefold sleeves "
        "is as well",
        "*was £295 27th jun '25 reduced 8th jun '25 £282 7th aug '25 £275 22nd aug "
        "'25 £269 5th sep '25 reduced 10th oct '25*",
        "superlow shipping prices to the europe and the us",
        "label variation",
        "orders usually processed within 24-48 hours",
        "in business since 1979",
    ]

    @pytest.mark.parametrize("raw", CORPUS)
    def test_corpus_promo_only_becomes_empty(self, preprocessor, raw):
        out = preprocessor.clean_text(raw)
        assert out.strip() == ""

    @pytest.mark.parametrize(
        "ship_tail",
        [
            " / $1.00 unlimited us-shipping / free on $9 orders of 3+ items read "
            "seller terms before paying",
            " / $99.99 unlimited us-shipping / free on $200 orders of 3+ items "
            "read seller terms before paying",
        ],
    )
    def test_us_shipping_tail_any_amounts(self, preprocessor, ship_tail):
        assert preprocessor.clean_text(ship_tail).strip() == ""

    def test_us_shipping_concatenated_with_markdown_still_empty(
        self, preprocessor
    ):
        combined = (
            "**all items sent securely in a double padded mailer with the vinyl "
            "separated from the sleeve (unless sealed)**"
            " / $5 unlimited us-shipping / free on $100 orders of 3+ items read "
            "seller terms before paying"
        )
        assert preprocessor.clean_text(combined).strip() == ""

    def test_star_buy2_get1_least_priced_note_block_removed(
        self, preprocessor
    ):
        block = (
            "**all $1 & $2 items = buy 2 get 1 free !! note: price deduction will "
            "be made on the least priced items in the order post-invoice so "
            "please refrain from making payment until final subtotal is adjusted**"
        )
        assert preprocessor.clean_text(block).strip() == ""
        alt = (
            "**all $10 & $20 items = buy 2 get 1 free !! note: price deduction will "
            "be made on the least priced items in the order post-invoice so "
            "please refrain from making payment until final subtotal is adjusted**"
        )
        assert preprocessor.clean_text(alt).strip() == ""
        mixed = (
            "vg+ sleeve, light wear. "
            "**all $3 & $4 items = buy 2 get 1 free !! note: price deduction will "
            "be made on the least priced items in the order post-invoice so "
            "please refrain from making payment until final subtotal is adjusted**"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "buy 2 get 1" not in out
        assert "post-invoice" not in out

    def test_priced_to_move_ships_quickly_storage_mailers_ctas_removed(
        self, preprocessor
    ):
        for raw in (
            "priced to move!",
            "need to create more storage space.",
            "ships quickly",
            "ships quickly, same or next day",
            "same or next day",
            "secure vinyl mailer",
            "pics available",
            "pics available upon request",
            "reach out w/ any questions",
            "reach out with any questions",
            "us orders only",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        combined = (
            "ships quickly, same or next day. secure vinyl mailers! "
            "us orders only reach out with any questions"
        )
        assert preprocessor.clean_text(combined).strip() == ""
        mixed = (
            "vg+ sleeve, light ring wear. "
            "priced to move. ships quickly, same or next day. "
            "secure vinyl mailer. pics available. us orders only."
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "ring wear" in out
        assert "priced" not in out
        assert "ships quickly" not in out
        assert "mailer" not in out
        assert "pics available" not in out
        assert "orders only" not in out

    def test_offer_for_euro_or_less_removed(self, preprocessor):
        for raw in (
            "offer for 2€ or less",
            "offers for 2 € or less",
            "offer for 1,50€ or less!",
            "offer for 3 eur or less",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        mixed = "vg+ sleeve. offer for 2€ or less."
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "offer for" not in out
        assert "or less" not in out

    def test_see_my_shipping_policy_mail_prices_removed(self, preprocessor):
        for raw in (
            "see my shipping policy for correct mail prices",
            "see my shipping policy for correct mail price!",
            "see my shipping policy for correct postal prices.",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        mixed = (
            "vg+ sleeve, light wear. "
            "see my shipping policy for correct mail prices."
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "shipping policy" not in out

    def test_get_pct_off_final_price_removed(self, preprocessor):
        for raw in (
            "get 20% off final price",
            "get 10 % off final price!",
            "get 5% off final price.",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        mixed = "vg+ sleeve. get 20% off final price when you buy 3."
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "final price" not in out
        assert "20%" not in out

    def test_protective_box_wisconsin_flat_domestic_thanks_small_business_removed(
        self, preprocessor
    ):
        for raw in (
            "ships in protective box",
            "ship in a protective boxes!",
            "independent start-up in wisconsin",
            "independent startup in wisconsin.",
            "independent start up in wisconsin",
            "flat rate $7.50 domestic shipping",
            "flat rate £4 domestic shipping!",
            "thanks for supporting small business",
            "thanks for supporting a small business.",
            "thanks for supporting small businesses!",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        mixed = (
            "vg+ sleeve, light wear. ships in protective box. "
            "flat rate $7.50 domestic shipping. "
            "thanks for supporting small business. "
            "independent start-up in wisconsin."
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "protective" not in out
        assert "wisconsin" not in out
        assert "domestic shipping" not in out
        assert "small business" not in out

    def test_grade_mint_only_when_still_sealed_policy_removed(self, preprocessor):
        raw = "i generally only grade mint when records are still sealed"
        assert preprocessor.clean_text(raw).strip() == ""
        assert preprocessor.clean_text(raw + ".").strip() == ""
        mixed = (
            "vg+ sleeve, light ring wear. "
            "i generally only grade mint when records are still sealed."
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "ring wear" in out
        assert "grade mint" not in out
        assert "still sealed" not in out

    def test_shipping_time_threshold_and_la_store_blurbs_removed(
        self, preprocessor
    ):
        for raw in (
            "free over $30",
            "free over $90!",
            "free over £25.",
            "usually same day",
            "usually same day!",
            "free same day international shipping",
            "average 8 - 10 days",
            "average 8–10 days.",
            "all shipping includes tracking",
            "the same price whether you buy 1 or 100 records",
            "the same price whether you buy 1 or 12 records!",
            "shipped from our l.a store",
            "shipped from our la store.",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        # Bare ``over $N`` is not stripped (would hit ``just over $90`` prose).
        keep = "vg+ sleeve. just over $90 in value on discogs."
        out_keep = preprocessor.clean_text(keep).strip()
        assert "just over" in out_keep
        assert "$90" in out_keep or "90" in out_keep
        mixed = (
            "nm vinyl, light hairlines. free over $30. usually same day. "
            "free same day international shipping. average 8 - 10 days. "
            "all shipping includes tracking. "
            "the same price whether you buy 1 or 100 records. "
            "shipped from our l.a store."
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "hairlines" in out
        assert "free over" not in out
        assert "same day" not in out
        assert "international shipping" not in out
        assert "average" not in out
        assert "includes tracking" not in out
        assert "same price" not in out
        assert "l.a" not in out
        assert "store" not in out

    def test_money_unlimited_items_shipped_in_us_removed(self, preprocessor):
        raw = "$7 unlimited items shipped in the u.s"
        assert preprocessor.clean_text(raw).strip() == ""
        assert preprocessor.clean_text(raw + ".").strip() == ""
        assert preprocessor.clean_text("$12 unlimited item shipped in the u.s!").strip() == ""
        mixed = "vg+ sleeve. $7 unlimited items shipped in the u.s."
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "unlimited" not in out
        assert "shipped in the" not in out

    def test_low_priced_worldwide_delivery_standalone_removed(self, preprocessor):
        for raw in (
            "low priced",
            "low-priced!",
            "worldwide delivery",
            "worldwide delivery.",
            "low priced quick worldwide delivery",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        mixed = "vg+ sleeve, light wear. low priced. worldwide delivery."
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "low priced" not in out
        assert "worldwide delivery" not in out

    def test_unlimited_uk_shipping_banner_and_photos_on_request_removed(
        self, preprocessor
    ):
        for raw in (
            "£5 unlimited uk shipping",
            "£ 5.50 unlimited uk shipping!",
            "€4 unlimited uk shipping",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        for raw in (
            "photos on request",
            "photo on request.",
            "photos upon request",
            "more photos upon request.",
            "more photos on request",
            "photos on demand!",
        ):
            assert preprocessor.clean_text(raw).strip() == ""
        sleeve = "vg+ sleeve, light wear. photos on the sleeve under raking light."
        assert "photos on the sleeve" in preprocessor.clean_text(sleeve).lower()
        mixed = (
            "nm vinyl. £5 unlimited uk shipping. more photos on request."
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "unlimited" not in out
        assert "photos on" not in out

    def test_mixed_condition_and_promo_tail_keeps_condition(self, preprocessor):
        raw = (
            "corner wear and ring wear on cover. "
            " / $6.40 unlimited us-shipping / free on $100 orders of 3+ items read "
            "seller terms before paying"
        )
        out = preprocessor.clean_text(raw)
        assert "corner wear" in out
        assert "unlimited" not in out
        assert "seller terms" not in out

    def test_uk_bulk_ship_promo_variants(self, preprocessor):
        raw = (
            "uk upto 8 records delivered 2nd class for £4.00 or 4 1st class for "
            "£4.50 free uk shipping on £50 or over orders"
        )
        assert preprocessor.clean_text(raw).strip() == ""
        alt = (
            "uk up to 12 records delivered 2nd class for £3.00 or 6 1st class for "
            "£5.00 free uk shipping on £100 or over order"
        )
        assert preprocessor.clean_text(alt).strip() == ""

    def test_post_multi_records_same_ship_price_promo_removed(
        self, preprocessor
    ):
        raw = "post x4 records for the same price as shipping one record"
        assert preprocessor.clean_text(raw).strip() == ""
        alt = "post x 3 records for the same price as shipping one record"
        assert preprocessor.clean_text(alt).strip() == ""
        mixed = (
            "vg+ sleeve, light wear. "
            "post x2 records for the same price as shipping one record"
        )
        out = preprocessor.clean_text(mixed)
        assert "very good plus" in out
        assert "post x" not in out
        assert "same price" not in out

    def test_shop_items_and_collect_cta_promos_removed(self, preprocessor):
        raw = "100 000+ items in our shop"
        assert preprocessor.clean_text(raw).strip() == ""
        raw2 = (
            "100 000+ items in our shop in upminster essex (district line / m25)"
        )
        assert preprocessor.clean_text(raw2).strip() == ""
        assert (
            preprocessor.clean_text("100000+ items in our shop").strip() == ""
        )
        assert (
            preprocessor.clean_text(
                "- 1000's more records & cds at our shop in upminster essex"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "- 1000s more records & cd's at our shop in upminster essex"
            ).strip()
            == ""
        )
        cta = "you can collect in store we buy records!"
        assert preprocessor.clean_text(cta).strip() == ""
        assert preprocessor.clean_text(cta[:-1]).strip() == ""  # no !
        mixed = (
            "nm vinyl, light hairlines. "
            "50 000+ items in our shop you can collect in store we buy records!"
        )
        out = preprocessor.clean_text(mixed)
        assert "near mint" in out
        assert "hairlines" in out
        assert "items in our shop" not in out
        assert "collect in store" not in out
        mixed_dash = (
            "vg+ sleeve. "
            "- 1000's more records & cds at our shop in upminster essex"
        )
        out_d = preprocessor.clean_text(mixed_dash).strip()
        assert "very good plus" in out_d
        assert "upminster" not in out_d
        assert "1000" not in out_d

    def test_degritter_ultrasonic_promo_removed(self, preprocessor):
        raw = (
            "cleaned in a degritter - the best ultrasonic record cleaning"
        )
        assert preprocessor.clean_text(raw).strip() == ""
        en_dash = (
            "cleaned in a degritter – the best ultrasonic record cleaning"
        )
        assert preprocessor.clean_text(en_dash).strip() == ""
        mixed = (
            "vg+ media, light marks. "
            "cleaned in a degritter - the best ultrasonic record cleaning"
        )
        out = preprocessor.clean_text(mixed)
        assert "very good plus" in out
        assert "degritter" not in out
        assert "ultrasonic" not in out

    def test_inventory_ultrasonic_cleaned_and_allow_week_order_status_removed(
        self, preprocessor
    ):
        inv = (
            "everything in our inventory is ultrasonically cleaned before shipment"
        )
        assert preprocessor.clean_text(inv).strip() == ""
        assert (
            preprocessor.clean_text(
                "everything in our inventory is ultrasonic cleaned prior to shipment."
            ).strip()
            == ""
        )
        status = (
            "please allow up to 1 week before checking the status of your order"
        )
        assert preprocessor.clean_text(status).strip() == ""
        assert (
            preprocessor.clean_text(
                "please allow up to 2 weeks before checking the status of your order!"
            ).strip()
            == ""
        )
        mixed = (
            "vg+ sleeve, light wear. "
            + inv
            + ". "
            + status
            + "."
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "inventory" not in out
        assert "ultrasonic" not in out
        assert "allow up to" not in out
        assert "status of your order" not in out

    def test_everything_pct_off_through_date_and_sealed_mm_policy_removed(
        self, preprocessor
    ):
        sale = "everything is 10% off through december 31"
        assert preprocessor.clean_text(sale).strip() == ""
        assert (
            preprocessor.clean_text(
                "everything is 15 % off through january 5th, 2026!"
            ).strip()
            == ""
        )
        sealed = (
            "this includes sealed items if the condition is m/m then it is sealed"
        )
        assert preprocessor.clean_text(sealed).strip() == ""
        assert (
            preprocessor.clean_text(
                "this includes sealed items if the condition is m / m then it is sealed."
            ).strip()
            == ""
        )
        mixed = f"vg+ sleeve, light wear. {sale}. {sealed}."
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "december" not in out
        assert "10%" not in out
        assert "sealed items if" not in out
        assert "then it is sealed" not in out

    def test_disc_anti_static_ultrasonic_vpi_whiplash_promos_removed(
        self, preprocessor
    ):
        assert (
            preprocessor.clean_text("disc stored in anti static inner").strip()
            == ""
        )
        assert preprocessor.clean_text("ultrasonic cleaned").strip() == ""
        assert preprocessor.clean_text("vpi vacuumed").strip() == ""
        assert (
            preprocessor.clean_text(
                "shipped in sturdy whiplash mailer"
            ).strip()
            == ""
        )
        assert preprocessor.clean_text("whiplash mailer").strip() == ""
        mixed = (
            "nm media, light hairlines. disc stored in anti static inner "
            "ultrasonic cleaned vpi vacuumed shipped in sturdy whiplash mailer"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "hairlines" in out
        assert "anti static" not in out
        assert "whiplash" not in out
        assert "vpi" not in out
        assert "ultrasonic cleaned" not in out

    def test_uk_shipping_and_marrs_shop_blurbs_removed(self, preprocessor):
        assert preprocessor.clean_text("all orders in uk sent first class").strip() == ""
        assert preprocessor.clean_text("recorded delivery").strip() == ""
        assert (
            preprocessor.clean_text("europe/worldwide with tracking").strip()
            == ""
        )
        assert preprocessor.clean_text("we only use quality mailers").strip() == ""
        assert preprocessor.clean_text("ship throughout the week").strip() == ""
        assert preprocessor.clean_text("we do this professionally").strip() == ""
        assert (
            preprocessor.clean_text(
                "postal charges reflect quality care and services used"
            ).strip()
            == ""
        )
        marrs = (
            "we're marrs plectrum records official rsd real world indie shop in "
            "peterborough uk"
        )
        assert preprocessor.clean_text(marrs).strip() == ""
        marrs_curly = (
            "we\u2019re marrs plectrum records official rsd real world indie shop "
            "in peterborough uk"
        )
        assert preprocessor.clean_text(marrs_curly).strip() == ""
        mixed = (
            "vg+ sleeve. all orders in uk sent first class "
            "europe/worldwide with tracking we only use quality mailers "
            "ship throughout the week"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "first class" not in out
        assert "tracking" not in out
        assert "quality mailers" not in out

    def test_pics_available_upon_request_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("pics available upon request").strip() == ""
        mixed = "nm vinyl, light marks. pics available upon request"
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "light marks" in out
        assert "pics available" not in out

    def test_cheapest_price_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("cheapest price").strip() == ""
        mixed = "vg+ sleeve, light wear. cheapest price on discogs"
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "cheapest price" not in out

    def test_paypal_pay_in_3_cleaning_video_promos_removed(self, preprocessor):
        assert preprocessor.clean_text("accepting paypal credit").strip() == ""
        assert preprocessor.clean_text("pay in 3").strip() == ""
        assert (
            preprocessor.clean_text("watch my cleaning process here").strip()
            == ""
        )
        mixed = (
            "vg+ media, light hairlines. accepting paypal credit pay in 3 "
            "watch my cleaning process here"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "hairlines" in out
        assert "paypal" not in out
        assert "pay in 3" not in out
        assert "cleaning process" not in out

    def test_orders_over_ship_for_threshold_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("orders over $60 ship for $6").strip() == ""
        assert (
            preprocessor.clean_text("orders over £50 ship for £4.50").strip()
            == ""
        )
        mixed = (
            "nm media, light marks. orders over $99.99 ship for $8 corner ding"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "corner ding" in out
        assert "orders over" not in out
        assert "ship for" not in out

    def test_free_shipping_usa_orders_over_threshold_removed(
        self, preprocessor
    ):
        assert (
            preprocessor.clean_text(
                "free shipping on usa orders over $30"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "free shipping on usa orders over $100.00"
            ).strip()
            == ""
        )
        mixed = (
            "vg+ sleeve, light ring wear. "
            "free shipping on usa orders over $45 small corner bump"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "ring wear" in out
        assert "corner bump" in out
        assert "free shipping on usa" not in out

    def test_flatrate_six_continents_promo_removed(self, preprocessor):
        assert (
            preprocessor.clean_text(
                "flatrate shipping rates to all 6 continents"
            ).strip()
            == ""
        )
        mixed = (
            "nm media, light marks. flatrate shipping rates to all 6 continents"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "continents" not in out

    def test_from_our_us_hub_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("from our us-hub").strip() == ""
        mixed = "vg+ sleeve, light wear. shipped from our us-hub today"
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "us-hub" not in out

    def test_check_other_black_sabbath_combine_shipping_removed(
        self, preprocessor
    ):
        raw = (
            "check my other black sabbath records and combine shipping !!!"
        )
        assert preprocessor.clean_text(raw).strip() == ""
        assert (
            preprocessor.clean_text(
                "check my other black sabbath records and combine shipping!!"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "check my other black sabbath records and combine shipping"
            ).strip()
            == ""
        )
        mixed = (
            "nm media, light marks. "
            "check my other black sabbath records and combine shipping !!!!"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "black sabbath" not in out
        assert "combine shipping" not in out

    def test_buy_n_cheapest_free_and_eu_free_shipping_promos_removed(
        self, preprocessor
    ):
        assert (
            preprocessor.clean_text(
                "buy 12 records and get the cheapest for free"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "buy 6 records and get the cheapest for free"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "you will receive 2 free records in the same style"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "you will receive 5 free records in the same style"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "free shipping: above 145 euro in europe (eu)"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "free shipping: above 200.50 euros in europe ( eu )"
            ).strip()
            == ""
        )
        mixed = (
            "vg+ sleeve, light wear. buy 3 records and get the cheapest for free "
            "you will receive 1 free records in the same style "
            "free shipping: above 99 euro in europe (eu) corner ding"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "corner ding" in out
        assert "cheapest for free" not in out
        assert "same style" not in out
        assert "europe (eu)" not in out

    def test_house_techno_stock_promo_removed(self, preprocessor):
        raw = (
            "check out our big stock of house techno trance disco & more"
        )
        assert preprocessor.clean_text(raw).strip() == ""
        mixed = (
            "nm media, light marks. "
            "check out our big stock of house techno trance disco & more"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "house techno" not in out

    def test_barcelona_pickup_and_worldwide_shipping_promos_removed(
        self, preprocessor
    ):
        assert preprocessor.clean_text("pick up in barcelona possible").strip() == ""
        assert (
            preprocessor.clean_text("cheap worldwide shipping price").strip()
            == ""
        )
        mixed = (
            "vg+ sleeve, light wear. pick up in barcelona possible "
            "cheap worldwide shipping price"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "barcelona" not in out
        assert "worldwide shipping" not in out

    def test_regular_lp_shipping_gatefold_promo_removed(self, preprocessor):
        raw = (
            "regular 1-5lp is the same shipping cost (2lp count as 2) gatefold "
            "sleeves is as well"
        )
        assert preprocessor.clean_text(raw).strip() == ""
        mixed = (
            "vg+ sleeve, light wear. regular 1-5lp is the same shipping cost "
            "(2lp count as 2) gatefold sleeves is as well corner bump"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "corner bump" in out
        assert "1-5lp" not in out
        assert "gatefold sleeves" not in out

    def test_star_was_price_reduction_history_removed(self, preprocessor):
        ascii_hist = (
            "*was £295 27th jun '25 reduced 8th jun '25 £282 7th aug '25 £275 "
            "22nd aug '25 £269 5th sep '25 reduced 10th oct '25*"
        )
        assert preprocessor.clean_text(ascii_hist).strip() == ""
        curly = (
            "*was £295 27th jun \u201825 reduced 8th jun \u201825 £282 7th aug\u2019 "
            "\u201825 £275 22nd aug \u201825 £269 5th sep \u201825 reduced "
            "10th oct \u201825*"
        )
        assert preprocessor.clean_text(curly).strip() == ""
        mixed = (
            "vg+ sleeve, light wear. "
            "*was $199 1st jan '24 reduced 2nd feb '24 $175* corner bump"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "corner bump" in out
        assert "was $" not in out
        assert "reduced" not in out

    def test_superlow_shipping_europe_us_promo_removed(self, preprocessor):
        raw = "superlow shipping prices to the europe and the us"
        assert preprocessor.clean_text(raw).strip() == ""
        mixed = (
            "nm media, light marks. superlow shipping prices to the europe and the us"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "superlow" not in out

    def test_label_variation_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("label variation").strip() == ""
        mixed = "vg+ sleeve, light wear. label variation noted on runout"
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "label variation" not in out

    def test_orders_processed_and_in_business_since_promos_removed(
        self, preprocessor
    ):
        assert (
            preprocessor.clean_text(
                "orders usually processed within 24-48 hours"
            ).strip()
            == ""
        )
        assert preprocessor.clean_text("in business since 1979").strip() == ""
        mixed = (
            "vg+ sleeve, light wear. orders usually processed within 24-48 hours "
            "in business since 1979"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "24-48" not in out
        assert "1979" not in out

    def test_sealed_one_way_as_is_no_returns_promo_removed(self, preprocessor):
        slash = (
            '/all sealed items are sold "one way / as is" and cannot be returned '
            "or exchanged"
        )
        assert preprocessor.clean_text(slash).strip() == ""
        no_slash = (
            'all sealed items are sold "one way / as is" and cannot be returned '
            "or exchanged"
        )
        assert preprocessor.clean_text(no_slash).strip() == ""
        mixed = (
            "vg+ sleeve, light ring wear. "
            '/all sealed items are sold "one way / as is" and cannot be returned '
            "or exchanged corner bump"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "ring wear" in out
        assert "corner bump" in out
        assert "one way" not in out
        assert "returned" not in out

    def test_international_buyers_shipping_quote_promo_removed(
        self, preprocessor
    ):
        raw = (
            "international buyers message me for your shipping quote"
        )
        assert preprocessor.clean_text(raw).strip() == ""
        with_comma = (
            "international buyers, message me for your shipping quote!"
        )
        assert preprocessor.clean_text(with_comma).strip() == ""
        mixed = (
            "nm vinyl, light marks. "
            "international buyers message me for your shipping quote"
        )
        out = preprocessor.clean_text(mixed)
        assert "near mint" in out
        assert "light marks" in out
        assert "shipping quote" not in out
        assert "international buyers" not in out

    def test_international_kg_parcel_and_fill_parcel_promos_removed(
        self, preprocessor
    ):
        assert (
            preprocessor.clean_text(
                "international shipping in 2 kg parcels"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "international shipping in 10 kg parcel"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "fill up your parcel with 6 records/ 15 cds"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "fill up your parcel with 3 records / 7 cd's"
            ).strip()
            == ""
        )
        mixed = (
            "nm vinyl, light marks. international shipping in 2 kg parcels "
            "hairline only"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "hairline" in out
        assert "kg parcels" not in out
        mixed_fill = (
            "vg+ sleeve. fill up your parcel with 6 records/ 15 cds ring wear"
        )
        out_f = preprocessor.clean_text(mixed_fill).strip()
        assert "very good plus" in out_f
        assert "ring wear" in out_f
        assert "fill up your parcel" not in out_f

    def test_ships_business_days_usps_tracking_within_us_custom_mailers_removed(
        self, preprocessor
    ):
        assert preprocessor.clean_text("ships in 2 business days").strip() == ""
        assert preprocessor.clean_text("ship in 1 business day.").strip() == ""
        assert preprocessor.clean_text("usps media mail").strip() == ""
        assert (
            preprocessor.clean_text("usps media mail with tracking").strip() == ""
        )
        assert preprocessor.clean_text("ships with tracking").strip() == ""
        assert preprocessor.clean_text("sent with tracking!").strip() == ""
        assert (
            preprocessor.clean_text("$6 shipping within the u.s").strip() == ""
        )
        assert (
            preprocessor.clean_text("£3 shipping within the us.").strip() == ""
        )
        long_blurb = (
            "i use custom shipping mailers with added corner protection and "
            "cardboard padding"
        )
        assert preprocessor.clean_text(long_blurb).strip() == ""
        mixed = (
            "vg+ sleeve, light wear. ships in 2 business days ring bump "
            "usps media mail $6 shipping within the u.s"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "very good plus" in out
        assert "ring bump" in out
        assert "business days" not in out
        assert "usps" not in out
        assert "shipping within" not in out

    def test_low_priced_quick_worldwide_delivery_removed(self, preprocessor):
        assert (
            preprocessor.clean_text(
                "low priced quick worldwide delivery"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text("low-priced quick worldwide delivery!")
            .strip()
            == ""
        )
        out = preprocessor.clean_text(
            "nm vinyl. low priced quick worldwide delivery light marks"
        ).strip()
        assert "near mint" in out
        assert "light marks" in out
        assert "worldwide delivery" not in out

    def test_qualify_free_shipping_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("qualify for free shipping").strip() == ""
        assert (
            preprocessor.clean_text("may qualify for free shipping").strip()
            == ""
        )
        assert (
            preprocessor.clean_text("orders qualify for free shipping").strip()
            == "orders"
        )
        mixed = (
            "vg+ sleeve, light ring wear. "
            "orders over $25 qualify for free shipping"
        )
        out = preprocessor.clean_text(mixed)
        assert "very good plus" in out
        assert "ring wear" in out
        assert "qualify" not in out
        assert "free shipping" not in out

    def test_misc_shop_shipping_promos_removed(self, preprocessor):
        assert preprocessor.clean_text("$9 flat shipping!").strip() == ""
        assert preprocessor.clean_text("£3.50 flat shipping!").strip() == ""
        assert (
            preprocessor.clean_text(
                "$6.50 flat rate shipping on all orders!"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "€4 flat rate shipping on all orders!"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "$6.50 flat rate shipping on all orders"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "$6.95 shipping in the u.s with tracking packed well and "
                "secure in a new"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "$6.95 shipping in the u.s with tracking packed well and "
                "secure in a new mailer."
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "$6.95 shipping in the u.s with tracking packed well and "
                "secure in a new whiplash mailer"
            ).strip()
            == ""
        )
        assert preprocessor.clean_text("cds ship in cardboard!").strip() == ""
        assert preprocessor.clean_text("cd's ship in cardboard!").strip() == ""
        assert preprocessor.clean_text("read seller terms!").strip() == ""
        assert (
            preprocessor.clean_text(
                "we're a real record store near pittsburgh pa"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "we\u2019re a real record store near pittsburgh pa"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "less than half of our inventory is posted here!"
            ).strip()
            == ""
        )
        assert (
            preprocessor.clean_text(
                "summer sale 10% off storewide 1 week only!"
            ).strip()
            == ""
        )
        mixed = (
            "vg+ sleeve, light wear. read seller terms! $9 flat shipping! "
            "cds ship in cardboard!"
        )
        out = preprocessor.clean_text(mixed)
        assert "very good plus" in out
        assert "read seller" not in out
        assert "flat shipping" not in out
        assert "cardboard" not in out
        mixed_flat_rate = (
            "ex sleeve. $6.50 flat rate shipping on all orders! small corner ding"
        )
        out_fr = preprocessor.clean_text(mixed_flat_rate).strip()
        assert "excellent" in out_fr
        assert "corner ding" in out_fr
        assert "flat rate" not in out_fr
        assert "all orders" not in out_fr
        mixed_us_ship = (
            "vg+ vinyl. $6.95 shipping in the u.s with tracking packed well "
            "and secure in a new. corner ding on sleeve"
        )
        out_us = preprocessor.clean_text(mixed_us_ship).strip()
        assert "very good plus" in out_us
        assert "corner ding" in out_us
        assert "tracking packed" not in out_us

    def test_buy_today_and_scoop_6x12_promos_removed(self, preprocessor):
        assert preprocessor.clean_text("buy this copy today").strip() == ""
        out = preprocessor.clean_text("nm copy buy this copy today").strip()
        assert "near mint" in out
        assert "buy this copy" not in out
        scoop = (
            'scoop purchase limited time buy 6x12" singles get 6x12" singles '
            "free ( cheapest free )"
        )
        assert preprocessor.clean_text(scoop).strip() == ""
        scoop_curly = (
            "scoop purchase limited time buy 6x12\u201d singles get 6x12"
            "\u201d singles free ( cheapest free )"
        )
        assert preprocessor.clean_text(scoop_curly).strip() == ""

    def test_uk_post_only_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("uk post only").strip() == ""
        assert preprocessor.clean_text("uk post only!").strip() == ""
        out = preprocessor.clean_text("vg+ sleeve. uk post only!").strip()
        assert "very good plus" in out
        assert "uk post" not in out

    def test_uk_pp_for_n_records_promo_removed(self, preprocessor):
        assert preprocessor.clean_text("uk p+p for 5 records").strip() == ""
        assert preprocessor.clean_text("uk p&p for 12 records").strip() == ""
        assert preprocessor.clean_text("uk p+p for 1 record").strip() == ""
        out = preprocessor.clean_text(
            "nm vinyl, light hairline. uk p+p for 5 records"
        ).strip()
        assert "near mint" in out
        assert "hairline" in out
        assert "p+p" not in out
        assert "p&p" not in out

    def test_jacksonville_fair_offers_star_and_unlimited_usa_ship_removed(
        self, preprocessor
    ):
        assert preprocessor.clean_text("jacksonville pressing").strip() == ""
        assert preprocessor.clean_text("all fair offers accepted").strip() == ""
        assert preprocessor.clean_text("⭐").strip() == ""
        assert preprocessor.clean_text("\u2b50\ufe0f").strip() == ""
        assert preprocessor.clean_text("$5 unlimited shipping in usa").strip() == ""
        assert (
            preprocessor.clean_text("£4 unlimited shipping in usa").strip() == ""
        )
        assert (
            preprocessor.clean_text("$5.00 unlimited shipping in usa").strip()
            == ""
        )
        mixed = (
            "⭐ nm media, light marks. jacksonville pressing "
            "$5 unlimited shipping in usa all fair offers accepted"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "light marks" in out
        assert "jacksonville" not in out
        assert "fair offers" not in out
        assert "\u2b50" not in out
        assert "unlimited shipping" not in out

    def test_starred_promos_and_uk_mainland_combine_removed(
        self, preprocessor
    ):
        assert preprocessor.clean_text("*buy 5 get cheapest free*").strip() == ""
        assert preprocessor.clean_text("* -accurate grading or refund*").strip() == ""
        uk = (
            'uk mainland customers 1-7 lp/12" or 30 7" singles same p&p '
            "combine & save*"
        )
        assert preprocessor.clean_text(uk).strip() == ""
        uk_curly = (
            "uk mainland customers 1-7 lp/12\u201d or 30 7\u201d singles "
            "same p&p combine & save*"
        )
        assert preprocessor.clean_text(uk_curly).strip() == ""
        mixed = (
            'nm vinyl. *buy 3 get cheapest free* '
            '* -accurate grading or refund* '
            'uk mainland customers 2-5 lp/10" or 12 7" singles same p&p '
            "combine & save* light marks"
        )
        out = preprocessor.clean_text(mixed).strip()
        assert "near mint" in out
        assert "light marks" in out
        assert "cheapest free" not in out
        assert "accurate grading" not in out
        assert "mainland customers" not in out

    def test_date_stamp_bracket_suffix_removed(self, preprocessor):
        assert (
            preprocessor.clean_text("still sealed 3/23]").strip()
            == "still sealed"
        )
        out = preprocessor.clean_text("corner wear 12/5] nm").strip()
        assert "12/5]" not in out
        assert "corner wear" in out
        assert "near mint" in out

    def test_double_star_condition_emphasis_not_stripped(self, preprocessor):
        """Sellers bold real defects with ** — must not delete protected terms."""
        raw = "**light stain** on cover, **seam split** at bottom edge"
        out = preprocessor.clean_text(raw)
        lost = preprocessor._verify_protected_terms(raw, out)
        assert "stain" not in lost
        assert "split" not in lost
        assert "seam split" not in lost
        assert "stain" in out
        assert "seam split" in out or "split" in out


class TestStripStrayNumericTokensFlag:
    def test_false_preserves_leading_catalog_digit(self, preprocessor, guidelines_path):
        cfg = preprocessor.config
        cfg["preprocessing"]["strip_stray_numeric_tokens"] = False
        pre = Preprocessor(
            config_path="unused",
            guidelines_path=guidelines_path,
            config=cfg,
        )
        assert "6" in pre.clean_text("6 sealed, new hype sticker").split()


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

    def test_verify_protected_terms_word_boundary(self, preprocessor):
        """Substring ``mark`` inside ``postmarked`` must not count as ``mark``."""
        assert "mark" in preprocessor.protected_terms
        raw = "postmarked corner, light mark on back"
        cleaned = "postmarked corner, light mark on back"
        lost = preprocessor._verify_protected_terms(raw, cleaned)
        assert "mark" not in lost
        lost_sub = preprocessor._verify_protected_terms(
            "postmarked only", "postmarked only"
        )
        assert "mark" not in lost_sub


class TestGatedStructuralPromoStripping:
    """Do not remove ``[]`` / ``###`` / ``***`` spans whose inner text matches a
    protected whole-token pattern (see ``strip_listing_promo_noise``)."""

    def test_bracket_with_scratch_preserved_in_clean_text(self, preprocessor):
        raw = "before [vg+ sleeve with light scratch] after"
        out = preprocessor.clean_text(raw)
        assert "scratch" in out
        assert "[" in out and "]" in out

    def test_hash_block_with_stain_preserved(self, preprocessor):
        raw = "### light stain on back cover ### rest of note"
        out = preprocessor.clean_text(raw)
        assert "stain" in out
        assert "###" in out

    def test_triple_star_promo_mixed_with_scratch_not_stripped(self, preprocessor):
        inner = (
            "free shipping on usa orders over $30 — "
            "light hairline scratch near run-out"
        )
        assert len(inner.strip()) >= 14
        raw = f"intro ***{inner}*** outro"
        out = preprocessor.clean_text(raw)
        assert "scratch" in out
        assert "***" in out

    def test_strip_listing_without_patterns_removes_bracket_with_scratch(self):
        phrases = load_promo_noise_patterns({})
        s = "before [light scratch on cover] after"
        out = strip_listing_promo_noise(s, phrases, protected_term_patterns=None)
        assert "scratch" not in out


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
