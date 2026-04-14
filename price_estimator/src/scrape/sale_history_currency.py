"""
Parse Discogs sale-history price strings and infer EUR→USD from data.

``price_original_text`` is the listing currency; ``price_user_currency_text`` is
Discogs' "your currency" (often EUR). We estimate **USD per EUR** from rows where
the listing is clearly **USD** and the user column is **EUR**, then scale user
amounts to approximate USD.
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable

# Leading symbols that are NOT US dollars.
_NON_USD_PREFIX = re.compile(
    r"^(A\$|AU\$|C\$|CA\$|CHF|NZ\$|R\$|£|€|¥)", re.IGNORECASE
)


def parse_usd_listing_amount(text: str | None) -> float | None:
    """
    Parse a **US dollar** listing price from ``price_original_text``.

    Returns ``None`` for ``£``, ``€``, ``A$``, ``CA$``, etc.
    """
    if not text:
        return None
    s = text.replace("\xa0", " ").strip()
    if not s or _NON_USD_PREFIX.match(s):
        return None
    if "€" in s and "$" not in s:
        return None
    if s.startswith("£"):
        return None
    m = re.search(r"(?<![A-Za-z])\$\s*([\d][\d.,]*)", s)
    if not m:
        return None
    raw = m.group(1)
    v = _to_float_money(raw)
    return v if v is not None and v > 0 else None


def parse_gbp_amount(text: str | None) -> float | None:
    """Parse a pound amount from ``price_original_text`` (``£``)."""
    if not text or "£" not in text:
        return None
    s = text.replace("\xa0", " ").strip()
    m = re.search(r"£\s*([\d][\d.,]*)", s)
    if not m:
        return None
    raw = m.group(1)
    v = _to_float_money(raw)
    return v if v is not None and v > 0 else None


def format_usd_money_string(amount: float) -> str:
    """``$`` + two decimals (e.g. ``$21.24``)."""
    if amount != amount:
        raise ValueError("amount must be finite")
    if amount < 0:
        raise ValueError("amount must be non-negative")
    return f"${float(amount):.2f}"


def parse_eur_amount(text: str | None) -> float | None:
    """Parse a Euro amount from a string containing ``€``."""
    if not text or "€" not in text:
        return None
    s = text.replace("\xa0", " ").strip()
    m = re.search(r"€\s*([\d][\d.,]*)", s)
    if not m:
        return None
    raw = m.group(1)
    v = _to_float_money(raw)
    return v if v is not None and v > 0 else None


def parse_usd_user_amount(text: str | None) -> float | None:
    """Parse US dollar amount from user column (already USD, no conversion)."""
    if not text:
        return None
    s = text.replace("\xa0", " ").strip()
    if "€" in s:
        return None
    if _NON_USD_PREFIX.match(s):
        return None
    m = re.search(r"(?<![A-Za-z])\$\s*([\d][\d.,]*)", s)
    if not m:
        return None
    raw = m.group(1)
    v = _to_float_money(raw)
    return v if v is not None and v > 0 else None


def _to_float_money(raw: str) -> float | None:
    t = raw.strip()
    if not t:
        return None
    if t.count(",") == 1 and t.count(".") == 0:
        t = t.replace(",", ".")
    elif t.count(".") == 1 and t.count(",") == 0:
        pass
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def usd_per_eur_from_pairs(
    ratios: Iterable[float],
    *,
    lo: float = 0.95,
    hi: float = 1.35,
) -> tuple[float | None, list[float]]:
    """
    Robust median of ``usd / eur`` with a sanity band (typical FX ~1.0–1.2).

    Returns ``(median_or_none, filtered_ratios_used)``.
    """
    xs = sorted(r for r in ratios if r == r and lo <= r <= hi)
    if not xs:
        return None, []
    return float(statistics.median(xs)), xs


def collect_usd_eur_ratios_from_rows(
    rows: Iterable[tuple[str, str]],
) -> list[float]:
    """Each row is ``(price_original_text, price_user_currency_text)``."""
    out: list[float] = []
    for orig, user in rows:
        usd = parse_usd_listing_amount(orig)
        eur = parse_eur_amount(user)
        if usd is None or eur is None or eur <= 0:
            continue
        out.append(usd / eur)
    return out
