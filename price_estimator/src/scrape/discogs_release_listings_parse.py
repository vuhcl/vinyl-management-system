"""
Parse Discogs **release marketplace** HTML (active listings).

Canonical URL shape (see ``collect_discogs_release_listings_botasaurus``)::

    https://www.discogs.com/sell/release/{release_id}?sort=price,asc&limit=250&page={n}

**Compliance:** Automated access to the Discogs website may be restricted by
Discogs terms of use. Operators must verify current policies before running
collectors in production.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup

from price_estimator.src.scrape.discogs_sale_history_parse import looks_like_login_or_challenge

_LISTING_ID_RE = re.compile(r"/sell/item/(\d+)", re.I)


def release_marketplace_url(
    release_id: str,
    *,
    page: int = 1,
    sort: str = "price,asc",
    limit: int = 250,
) -> str:
    """Build marketplace listing URL for a specific release and page."""
    rid = str(release_id).strip()
    p = max(1, int(page))
    lim = max(1, min(int(limit), 250))
    # Match Discogs links: comma in sort value is percent-encoded (e.g. price%2Casc).
    sort_q = quote(str(sort).strip(), safe="")
    return (
        f"https://www.discogs.com/sell/release/{rid}"
        f"?sort={sort_q}&limit={lim}&page={p}"
    )


def _first_listing_id(href: str | None) -> str | None:
    if not href:
        return None
    m = _LISTING_ID_RE.search(href)
    return m.group(1) if m else None


def _parse_media_sleeve_from_condition_block(text: str) -> tuple[str, str]:
    """
    Parse ``p.item_condition`` body: lines like ``Media: ...`` / ``Sleeve: ...``.
    Returns (raw_media, raw_sleeve) — may be empty if not in Discogs format.
    """
    raw_media = ""
    raw_sleeve = ""
    if not text or not text.strip():
        return raw_media, raw_sleeve
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("media:"):
            raw_media = line.split(":", 1)[1].strip()
        elif low.startswith("sleeve:"):
            raw_sleeve = line.split(":", 1)[1].strip()
    return raw_media, raw_sleeve


def _artist_title_from_item_line(title_text: str) -> tuple[str, str]:
    t = (title_text or "").strip()
    if " - " in t:
        artist, title = t.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", t


def _listing_dict_from_row(
    tr: Any,
    *,
    release_id: str,
) -> dict[str, Any] | None:
    td = tr.select_one("td.item_description")
    if td is None:
        return None

    a = td.select_one("strong a[href*='/sell/item/']")
    if a is None:
        a = td.select_one("a[href*='/sell/item/']")
    href = a.get("href") if a else None
    lid = _first_listing_id(href)
    if not lid:
        return None

    title_text = a.get_text(" ", strip=True) if a else ""
    artist, title = _artist_title_from_item_line(title_text)

    cond_el = td.select_one("p.item_condition")
    cond_text = cond_el.get_text("\n", strip=True) if cond_el else ""
    raw_media, raw_sleeve = _parse_media_sleeve_from_condition_block(cond_text)

    comment_parts: list[str] = []
    for p in td.find_all("p"):
        cls = " ".join(p.get("class") or [])
        if "item_condition" in cls:
            continue
        txt = p.get_text(" ", strip=True)
        if txt:
            comment_parts.append(txt)
    comments = "\n\n".join(comment_parts).strip()

    return {
        "id": int(lid) if lid.isdigit() else lid,
        "sleeve_condition": raw_sleeve,
        "condition": raw_media,
        "comments": comments,
        "release": {
            "artist": artist,
            "title": title,
            "year": None,
            "country": "",
            "format": "",
            "description": "",
            "_release_page_id": str(release_id).strip(),
        },
    }


@dataclass
class ParsedReleaseListingsPage:
    release_id: str
    page: int
    listings: list[dict[str, Any]]
    parse_warnings: list[str] = field(default_factory=list)


def parse_release_listings_html(
    html: str,
    release_id: str,
    *,
    page: int = 1,
) -> ParsedReleaseListingsPage:
    """
    Parse one marketplace results page into Discogs **inventory-shaped**
    listing dicts (compatible with ``DiscogsIngester.parse_listing``).
    """
    warnings: list[str] = []
    rid = str(release_id).strip()
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("#pjax_container") or soup.select_one("#page_content") or soup

    rows = root.select("table tbody tr")
    if not rows:
        for table in root.find_all("table"):
            rows = table.select("tbody tr")
            if rows:
                break

    listings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for tr in rows:
        if tr.select_one("th"):
            continue
        d = _listing_dict_from_row(tr, release_id=rid)
        if not d:
            continue
        iid = str(d.get("id", ""))
        if iid in seen_ids:
            continue
        seen_ids.add(iid)
        listings.append(d)

    if not listings:
        warnings.append("no_listing_rows_matched")

    return ParsedReleaseListingsPage(
        release_id=rid,
        page=int(page),
        listings=listings,
        parse_warnings=warnings,
    )


__all__ = [
    "ParsedReleaseListingsPage",
    "looks_like_login_or_challenge",
    "parse_release_listings_html",
    "release_marketplace_url",
]
