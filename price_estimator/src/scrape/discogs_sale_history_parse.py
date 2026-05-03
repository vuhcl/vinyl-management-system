"""
Parse Discogs logged-in Sales History HTML.

URL: ``https://www.discogs.com/sell/history/{release_id}``

**Compliance:** Automated access to the Discogs website may be restricted by
Discogs terms of use. Operators must verify current policies before running
collectors in production (see ``collect_sale_history_botasaurus`` module doc).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Headers normalized via ``_norm_header`` — lookup order prefers explicit wording
# so we never attach the jacket column to ``media_condition`` or vice versa
# when Discogs revises labels (some locales use ``Media Condition`` only).
_MEDIA_HEADER_KEYS: tuple[str, ...] = (
    "media condition",
    "vinyl condition",
    "record condition",
    "vinyl grading",
    "record grading",
    "disc condition",
    # Legacy single column name on English sales history tables (vinyl/record).
    "condition",
)
_SLEEVE_HEADER_KEYS: tuple[str, ...] = (
    "sleeve condition",
    "jacket condition",
    "cover condition",
    "cover grading",
    "sleeve grading",
)


def _first_nonempty_header_cell(
    hdr: dict[str, int],
    row_cells: list[str],
    normalized_keys: tuple[str, ...],
) -> str:
    for nk in normalized_keys:
        ix = hdr.get(nk)
        if ix is None or ix >= len(row_cells):
            continue
        val = row_cells[ix].strip()
        if val:
            return val
    return ""


def sale_history_url(release_id: str) -> str:
    rid = str(release_id).strip()
    return f"https://www.discogs.com/sell/history/{rid}"


@dataclass
class SaleHistorySummary:
    last_sold_on: str | None = None
    average: float | None = None
    median: float | None = None
    high: float | None = None
    low: float | None = None
    raw_lines: list[str] = field(default_factory=list)


@dataclass
class SaleHistoryRow:
    order_date: str
    media_condition: str
    sleeve_condition: str
    price_original_text: str
    price_user_currency_text: str
    seller_comments: str

    def row_hash(self, release_id: str) -> str:
        key = "|".join(
            [
                release_id.strip(),
                self.order_date,
                self.media_condition,
                self.sleeve_condition,
                self.price_original_text,
                self.seller_comments,
            ]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class ParsedSaleHistory:
    release_id: str
    summary: SaleHistorySummary | None
    rows: list[SaleHistoryRow]
    parse_warnings: list[str] = field(default_factory=list)


def _strip_money(s: str) -> float | None:
    t = (s or "").strip()
    if not t:
        return None
    t = re.sub(r"[^\d.,-]", "", t.replace(",", ""))
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _parse_summary_from_text(text: str) -> SaleHistorySummary:
    out = SaleHistorySummary()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out.raw_lines = lines[:20]
    for ln in lines:
        low = ln.lower()
        if low.startswith("last sold on"):
            m = re.search(r"last sold on:?\s*(.+)$", ln, re.I)
            if m:
                out.last_sold_on = m.group(1).strip()
        elif low.startswith("average"):
            m = re.search(r"average:?\s*(.+)$", ln, re.I)
            if m:
                out.average = _strip_money(m.group(1))
        elif low.startswith("median"):
            m = re.search(r"median:?\s*(.+)$", ln, re.I)
            if m:
                out.median = _strip_money(m.group(1))
        elif low.startswith("high"):
            m = re.search(r"high:?\s*(.+)$", ln, re.I)
            if m:
                out.high = _strip_money(m.group(1))
        elif low.startswith("low"):
            m = re.search(r"low:?\s*(.+)$", ln, re.I)
            if m:
                out.low = _strip_money(m.group(1))
    return out


def _parse_summary(soup: BeautifulSoup) -> SaleHistorySummary | None:
    main = soup.select_one("#page_content, main, #pjax_container, body")
    if not main:
        return None
    return _parse_summary_from_text(main.get_text("\n", strip=True))


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip().lower())


def _header_map(header_cells: list[str]) -> dict[str, int]:
    return {_norm_header(h): i for i, h in enumerate(header_cells)}


def _table_if_sale_history(table: Any) -> Any | None:
    headers = table.find_all("th")
    if not headers:
        return None
    texts = [h.get_text(" ", strip=True) for h in headers]
    joined = " ".join(t.lower() for t in texts)
    if "order date" in joined and "condition" in joined and "price" in joined:
        return table
    return None


def _find_sales_table(soup: BeautifulSoup) -> Any | None:
    # Live Discogs uses e.g. <table class="... sales-history-table">.
    pinpoint = soup.select_one("table.sales-history-table")
    if pinpoint is not None:
        hit = _table_if_sale_history(pinpoint)
        if hit is not None:
            return hit
    for table in soup.find_all("table"):
        if table is pinpoint:
            continue
        hit = _table_if_sale_history(table)
        if hit is not None:
            return hit
    return None


def _parse_table(table: Any, warnings: list[str]) -> list[SaleHistoryRow]:
    # Include ``thead`` rows so column indices are known before ``tbody`` data.
    rows: list[SaleHistoryRow] = []
    header_idx: dict[str, int] | None = None

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]

        if any(c.name == "th" for c in cells) or (
            len(texts) >= 4
            and texts[0].lower() in ("order date", "date")
        ):
            header_idx = _header_map(texts)
            continue

        if header_idx is None:
            continue

        hdr = header_idx

        def cell(exact_key: str) -> str:
            ix = hdr.get(exact_key)
            if ix is None or ix >= len(texts):
                return ""
            return texts[ix]

        def cell_sub(needle: str) -> str:
            for k, ix in hdr.items():
                if needle in k:
                    return texts[ix] if ix < len(texts) else ""
            return ""

        od = texts[0] if texts else ""
        is_data = bool(od and _DATE_RE.match(od))
        if not is_data:
            # Sub-row (e.g. seller comments) attached to the previous sale
            if rows:
                msg = " ".join(t for t in texts if t).strip()
                if msg:
                    prev = rows[-1]
                    merged = (
                        f"{prev.seller_comments} {msg}".strip()
                        if prev.seller_comments
                        else msg
                    )
                    rows[-1] = SaleHistoryRow(
                        order_date=prev.order_date,
                        media_condition=prev.media_condition,
                        sleeve_condition=prev.sleeve_condition,
                        price_original_text=prev.price_original_text,
                        price_user_currency_text=prev.price_user_currency_text,
                        seller_comments=merged,
                    )
            continue

        media = _first_nonempty_header_cell(hdr, texts, _MEDIA_HEADER_KEYS)
        sleeve = _first_nonempty_header_cell(hdr, texts, _SLEEVE_HEADER_KEYS)
        p_user = cell_sub("your currency") or cell_sub("price in your")
        p_orig = cell("price")
        if not p_orig:
            for t in reversed(texts):
                if t and t != p_user and any(
                    sym in t for sym in ("$", "€", "£", "¥", "CA$", "A$")
                ):
                    p_orig = t
                    break

        rows.append(
            SaleHistoryRow(
                order_date=od,
                media_condition=media,
                sleeve_condition=sleeve,
                price_original_text=p_orig,
                price_user_currency_text=p_user,
                seller_comments="",
            )
        )

    if not rows:
        warnings.append("no_table_rows_matched")
    return rows


def looks_like_login_or_challenge(html: str) -> bool:
    """Heuristic: page is not the sale history content."""
    low = html.lower()
    if "sign in" in low and "password" in low:
        return True
    if "just a moment" in low and "challenge" in low:
        return True
    if "enable javascript and cookies" in low:
        return True
    return False


def parse_sale_history_html(html: str, release_id: str) -> ParsedSaleHistory:
    warnings: list[str] = []
    rid = str(release_id).strip()
    soup = BeautifulSoup(html, "html.parser")
    summary = _parse_summary(soup)
    table = _find_sales_table(soup)
    if table is None:
        warnings.append("sales_table_not_found")
        rows: list[SaleHistoryRow] = []
    else:
        rows = _parse_table(table, warnings)
    return ParsedSaleHistory(
        release_id=rid,
        summary=summary,
        rows=rows,
        parse_warnings=warnings,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
