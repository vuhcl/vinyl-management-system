"""Tests for Discogs release marketplace HTML parsing."""

from __future__ import annotations

from price_estimator.src.scrape.discogs_release_listings_parse import (
    parse_release_listings_html,
    release_marketplace_url,
)
from price_estimator.src.scrape.discogs_sale_history_parse import looks_like_login_or_challenge

FIXTURE_HTML = """
<div id="pjax_container">
<table><tbody>
<tr>
  <td class="item_description">
    <strong><a href="/sell/item/999001/release/37091274">The Beatles - Abbey Road</a></strong>
    <p class="item_condition">Media: Near Mint (NM or M-)<br/>Sleeve: Very Good (VG)</p>
    <p>Small corner bump on jacket. Plays cleanly with faint surface noise.</p>
  </td>
  <td class="item_price hide_mobile"><span class="price">$25.00</span></td>
</tr>
<tr>
  <td class="item_description">
    <strong><a href="/sell/item/999002/release/37091274">The Beatles - Abbey Road</a></strong>
    <p class="item_condition">Media: Very Good Plus (VG+)<br/>Sleeve: Near Mint (NM or M-)</p>
  </td>
</tr>
</tbody></table>
</div>
"""


def test_release_marketplace_url() -> None:
    u = release_marketplace_url("37091274", page=1, limit=250, sort="price,asc")
    assert "37091274" in u
    assert "limit=250" in u
    assert "page=1" in u
    assert "sort=price%2Casc" in u


def test_parse_release_listings_fixture() -> None:
    p = parse_release_listings_html(FIXTURE_HTML, "37091274", page=1)
    assert p.release_id == "37091274"
    assert p.page == 1
    assert len(p.listings) == 2
    a, b = p.listings[0], p.listings[1]
    assert a["id"] == 999001
    assert a["condition"] == "Near Mint (NM or M-)"
    assert a["sleeve_condition"] == "Very Good (VG)"
    assert "corner bump" in (a.get("comments") or "").lower()
    assert a["release"]["artist"] == "The Beatles"
    assert a["release"]["title"] == "Abbey Road"
    assert b["id"] == 999002
    assert b["comments"] == ""


def test_parse_jacket_synonym_maps_to_sleeve_field() -> None:
    html = (
        '<div id="pjax_container"><table><tbody><tr>'
        '<td class="item_description">'
        '<strong><a href="/sell/item/999003/release/1">X - Y</a></strong>'
        '<p class="item_condition">Media: Near Mint (NM or M-)<br/>'
        "Jacket: Good (G)</p></td></tr></tbody></table></div>"
    )
    p = parse_release_listings_html(html, "1", page=1)
    assert len(p.listings) == 1
    assert p.listings[0]["condition"] == "Near Mint (NM or M-)"
    assert p.listings[0]["sleeve_condition"] == "Good (G)"


def test_looks_like_login_fixture() -> None:
    assert looks_like_login_or_challenge(FIXTURE_HTML) is False
