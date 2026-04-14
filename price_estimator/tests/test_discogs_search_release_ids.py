"""Tests for Discogs search release ID extraction."""
from __future__ import annotations

import pytest

from price_estimator.src.scrape.discogs_search_release_ids import (
    build_vinyl_decade_search_url,
    build_vinyl_decade_year_sort_search_url,
    extract_release_ids_from_html,
    iter_years_for_decade,
    js_collect_release_ids,
    release_id_from_href,
)


def test_build_vinyl_decade_search_url() -> None:
    u = build_vinyl_decade_search_url(decade=2020, page=3)
    assert u.startswith("https://www.discogs.com/search?")
    assert "type=release" in u
    assert "page=3" in u
    assert "format_exact=Vinyl" in u
    assert "decade=2020" in u


def test_iter_years_for_decade_full_and_capped() -> None:
    assert iter_years_for_decade(1990, through_year=1999) == tuple(range(1990, 2000))
    assert iter_years_for_decade(2020, through_year=2026) == tuple(range(2020, 2027))


def test_iter_years_for_decade_empty_future() -> None:
    assert iter_years_for_decade(2030, through_year=2026) == ()


def test_build_vinyl_decade_year_sort_search_url() -> None:
    u = build_vinyl_decade_year_sort_search_url(
        decade=2020, year=2024, page=2, sort_mode="have"
    )
    assert "decade=2020" in u
    assert "year=2024" in u
    assert "page=2" in u
    assert "sort=have%2Cdesc" in u or "sort=have,desc" in u

    r = build_vinyl_decade_year_sort_search_url(
        decade=2020, year=2025, page=1, sort_mode="relevance"
    )
    assert "year=2025" in r
    assert "sort=" not in r


def test_build_vinyl_decade_year_sort_invalid() -> None:
    with pytest.raises(ValueError):
        build_vinyl_decade_year_sort_search_url(
            decade=2020, year=2020, page=1, sort_mode="bogus"
        )


def test_release_id_from_href_real_card() -> None:
    assert release_id_from_href("/release/20159026-Fred-Hush-Secret-002") == "20159026"


def test_release_id_from_absolute_discogs_url() -> None:
    assert (
        release_id_from_href(
            "https://www.discogs.com/release/20159026-Fred-Hush-Secret-002"
        )
        == "20159026"
    )


def test_release_id_protocol_relative() -> None:
    assert release_id_from_href("//www.discogs.com/release/42-X") == "42"


def test_release_id_from_href_rejects_master() -> None:
    assert release_id_from_href("/master/123-Title") is None


def test_extract_release_ids_fixture_order_and_dedupe() -> None:
    html = """
    <html><body>
    <aside><a href="/release/999-Sidebar">x</a></aside>
    <div id="search_results">
      <a class="group absolute inset-0" href="/release/20159026-Fred-Hush-Secret-002">A</a>
      <a href="https://www.discogs.com/release/20159026-Fred-Hush-Secret-002">dup</a>
      <a href="/release/42-Other">B</a>
    </div>
    </body></html>
    """
    ids = extract_release_ids_from_html(html, scope_selector="#search_results")
    assert ids == ["20159026", "42"]


def test_extract_falls_back_to_body_when_scope_empty() -> None:
    html = """
    <div id="search_results"></div>
    <a href="/release/7-X">only</a>
    """
    ids = extract_release_ids_from_html(html, scope_selector="#search_results")
    assert ids == ["7"]


def test_js_collect_contains_regex_and_scope() -> None:
    src = js_collect_release_ids(scope_selector="#search_results")
    assert "#search_results" in src
    assert "/release/" in src
    assert "querySelector" in src


def test_js_collect_null_scope() -> None:
    src = js_collect_release_ids(scope_selector=None)
    assert "null" in src


def test_extract_release_ids_markup_fallback_no_anchors() -> None:
    html = """
    <html><body><div id="__next">
    <script type="application/json">{"uri":"/release/7777777-Slug"}</script>
    </div></body></html>
    """
    ids = extract_release_ids_from_html(html, scope_selector="#search_results")
    assert ids == ["7777777"]
