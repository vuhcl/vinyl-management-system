"""Seller comment normalization (URLs, emoji, whitespace)."""
from __future__ import annotations

import re

# Emoji / pictograph / common dingbat decoration (before boilerplate rules).
_EMOJI_AND_PICTO_RE = re.compile(
    (
        "["
        "\U0001F1E0-\U0001F1FF"  # regional indicator / flags
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"  # dingbats
        "\U00002600-\U000026FF"  # misc symbols (★, ☮, …)
        "]+"
    )
)


def normalize_seller_comment_text(
    text: str,
    *,
    strip_urls: bool = True,
    strip_emoji: bool = True,
) -> str:
    """
    Run before :meth:`DiscogsIngester.strip_boilerplate_from_notes` — collapse
    ``___`` underlines, ``http(s)`` and ``www.`` URLs, the ``links:`` label, and
    emoji / pictographic symbols. Whitespace is collapsed; empty string if input is
    empty after trim.
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"_{3,}", " ", s)
    if strip_urls:
        s = re.sub(r"(?i)\blinks:\s*", " ", s)
        s = re.sub(r"https?://\S+", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"\bwww\.\S+", " ", s, flags=re.IGNORECASE)
    if strip_emoji:
        s = _EMOJI_AND_PICTO_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
