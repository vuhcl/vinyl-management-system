"""
Condition features: encode sleeve and media condition (one-hot or ordinal).
Integrates with NLP classifier output (predicted condition) when available.
"""
from typing import Literal

import pandas as pd


# Discogs-style condition grades (worst to best for ordinal)
CONDITION_GRADES = [
    "Good",
    "Very Good",
    "Very Good Plus",
    "Near Mint",
    "Mint",
]


def _normalize_condition(s: str | None) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    for grade in CONDITION_GRADES:
        if grade.lower() == s.lower():
            return grade
    return None


def encode_condition_features(
    df: pd.DataFrame,
    *,
    sleeve_col: str = "sleeve_condition",
    media_col: str = "media_condition",
    encode: Literal["one_hot", "ordinal"] = "one_hot",
    prefix: str = "cond",
) -> pd.DataFrame:
    """
    Encode sleeve and media condition. Adds columns to a copy of df.

    - one_hot: one column per grade per sleeve/media (e.g. cond_sleeve_Near_Mint)
    - ordinal: single numeric column per sleeve/media (0=Good .. 4=Mint)
    """
    out = df.copy()
    for col, name in [(sleeve_col, "sleeve"), (media_col, "media")]:
        if col not in out.columns:
            continue
        normalized = out[col].map(_normalize_condition)
        if encode == "ordinal":
            grade_to_num = {g: i for i, g in enumerate(CONDITION_GRADES)}
            out[f"{prefix}_{name}_ord"] = (
                normalized.map(grade_to_num).fillna(-1).astype(int)
            )
        else:
            for grade in CONDITION_GRADES:
                key = f"{prefix}_{name}_{grade.replace(' ', '_')}"
                out[key] = (normalized == grade).astype(int)
    return out
