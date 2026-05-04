"""Tests for sale history currency parsing and EUR→USD calibration."""
from __future__ import annotations

import pytest

from price_estimator.src.scrape.sale_history_currency import (
    collect_usd_eur_ratios_from_rows,
    format_usd_money_string,
    parse_eur_amount,
    parse_gbp_amount,
    parse_loose_money_amount,
    parse_usd_listing_amount,
    parse_usd_user_amount,
    usd_per_eur_from_pairs,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$24.99", 24.99),
        ("US $10.00", 10.0),
        (" $3.50 ", 3.5),
    ],
)
def test_parse_usd_listing_amount_positive(text: str, expected: float) -> None:
    assert parse_usd_listing_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["£20.00", "€5.00", "A$23.00", "CA$15.00"],
)
def test_parse_usd_listing_amount_rejects_non_usd(text: str) -> None:
    assert parse_usd_listing_amount(text) is None


def test_parse_usd_listing_none_when_ambiguous() -> None:
    assert parse_usd_listing_amount("") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1,234.56", 1234.56),
        ("12,34", 12.34),
        ("  Average: $10.50 ", 10.5),
        ("", None),
    ],
)
def test_parse_loose_money_amount(text: str, expected: float | None) -> None:
    if expected is None:
        assert parse_loose_money_amount(text) is None
    else:
        assert parse_loose_money_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("€21.24", 21.24),
        ("€2.01", 2.01),
        ("Price €10", 10.0),
        ("no euro", None),
    ],
)
def test_parse_eur_amount(text: str, expected: float | None) -> None:
    if expected is None:
        assert parse_eur_amount(text) is None
    else:
        assert parse_eur_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$48.99", 48.99),
        ("$48.99 ", 48.99),
        ("€21.24", None),
    ],
)
def test_parse_usd_user_amount(text: str, expected: float | None) -> None:
    if expected is None:
        assert parse_usd_user_amount(text) is None
    else:
        assert parse_usd_user_amount(text) == pytest.approx(expected)


def test_collect_ratios_and_median() -> None:
    rows = [
        ("$24.99", "€21.24"),
        ("$10.00", "€8.50"),
        ("£1.00", "€1.15"),
    ]
    ratios = collect_usd_eur_ratios_from_rows(rows)
    assert len(ratios) == 2
    med, used = usd_per_eur_from_pairs(ratios, lo=1.0, hi=1.3)
    assert med is not None
    assert len(used) == 2
    assert med == pytest.approx(24.99 / 21.24, rel=1e-3)


def test_usd_per_eur_empty() -> None:
    med, used = usd_per_eur_from_pairs([], lo=0.9, hi=1.4)
    assert med is None
    assert used == []


def test_parse_gbp_amount() -> None:
    assert parse_gbp_amount("£20.00") == pytest.approx(20.0)
    assert parse_gbp_amount("£1.49") == pytest.approx(1.49)
    assert parse_gbp_amount("$10") is None


def test_format_usd_money_string() -> None:
    assert format_usd_money_string(21.24) == "$21.24"
    assert format_usd_money_string(27.030225) == "$27.03"
