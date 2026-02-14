"""
Text cleaning and preprocessing for seller notes.

- Lowercasing, URL removal, whitespace normalization
- Optional min/max length and token count filters
"""
import re
from pathlib import Path
from typing import Any

import pandas as pd


def clean_seller_notes(
    text: str | None,
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    strip_whitespace: bool = True,
    max_length_chars: int = 2000,
) -> str:
    """
    Clean a single seller note string.
    Returns empty string for None or non-string.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip()
    if not s:
        return ""
    if strip_whitespace:
        s = " ".join(s.split())
    if remove_urls:
        s = re.sub(r"https?://\S+", "", s, flags=re.IGNORECASE)
        s = " ".join(s.split())
    if lowercase:
        s = s.lower()
    if max_length_chars and len(s) > max_length_chars:
        s = s[:max_length_chars]
    return s


def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = "seller_notes",
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    strip_whitespace: bool = True,
    max_length_chars: int = 2000,
    min_tokens: int = 2,
) -> pd.DataFrame:
    """
    Preprocess a labeled dataset: clean text and optionally drop rows with too few tokens.
    """
    out = df.copy()
    if text_column not in out.columns:
        return out
    out["cleaned_notes"] = out[text_column].map(
        lambda x: clean_seller_notes(
            x,
            lowercase=lowercase,
            remove_urls=remove_urls,
            strip_whitespace=strip_whitespace,
            max_length_chars=max_length_chars,
        )
    )
    if min_tokens > 0:
        token_count = out["cleaned_notes"].str.split().str.len()
        out = out[token_count >= min_tokens].copy()
    return out


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load base config YAML for preprocessing and pipeline defaults."""
    import yaml
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}
