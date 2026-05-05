"""
grader/src/data/preprocess

Text preprocessing pipeline for the vinyl condition grader.
Reads unified.jsonl, applies text normalization and abbreviation
expansion, detects unverified media signals, assigns train/val/test
splits using adaptive stratification, and writes output JSONL files.

Transformation order (strictly enforced):
  1. Detect unverified media signals  — on raw text
  2. Detect Generic sleeve signals    — on raw text
  3. Lowercase
  4. Normalize whitespace
  5. Strip listing promo / shipping boilerplate (markdown, brackets, regex
     templates, configured phrase chunks) — on lowercased collapsed text.
     When protected-term patterns are supplied, ``###…###``, ``[…]``, and
     promo-gated ``***…***`` spans whose inner text matches a protected
     whole-token pattern are left intact to avoid dropping real defects.
  6. Optionally strip leading catalog digit before condition words
     (``strip_stray_numeric_tokens``)
  7. Expand abbreviations             — after lowercase
  8. Verify protected terms survive   — sanity check

The original `text` field is preserved. Cleaned text is written
to a new `text_clean` field. Labels are never modified.

Usage:
    python -m grader.src.data.preprocess
    python -m grader.src.data.preprocess --dry-run
"""

from __future__ import annotations

from .listing_promo import (
    build_protected_term_token_patterns,
    load_promo_noise_patterns,
    strip_listing_promo_noise,
)
from .preprocessor_core import Preprocessor

__all__ = [
    "Preprocessor",
    "build_protected_term_token_patterns",
    "load_promo_noise_patterns",
    "strip_listing_promo_noise",
]

