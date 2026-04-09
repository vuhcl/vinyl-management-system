"""
VinylIQ feature assembly: ordinal conditions + catalog + marketplace stats.

Used by training and inference. Genre is passed through as string; training
fits a target encoder saved in artifacts.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

# Plan: M=8, NM=7, VG+=6, VG=5, G+=4, G=3, F=2, P=1; sleeve extras
_CONDITION_MAP: dict[str, float] = {
    "mint (m)": 8.0,
    "mint": 8.0,
    "m": 8.0,
    "near mint (nm or m-)": 7.0,
    "near mint": 7.0,
    "nm": 7.0,
    "m-": 7.0,
    "very good plus (vg+)": 6.0,
    "very good plus": 6.0,
    "vg+": 6.0,
    "very good (vg)": 5.0,
    "very good": 5.0,
    "vg": 5.0,
    "good plus (g+)": 4.0,
    "good plus": 4.0,
    "g+": 4.0,
    "good (g)": 3.0,
    "good": 3.0,
    "g": 3.0,
    "fair (f)": 2.0,
    "fair": 2.0,
    "f": 2.0,
    "poor (p)": 1.0,
    "poor": 1.0,
    "p": 1.0,
    "generic": 0.0,
    "not graded": -1.0,
    "no cover": -2.0,
}


def condition_string_to_ordinal(label: str | None) -> float:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return -1.0
    s = str(label).strip().lower()
    if not s:
        return -1.0
    if s in _CONDITION_MAP:
        return _CONDITION_MAP[s]
    for key, val in _CONDITION_MAP.items():
        if key in s or s in key:
            return val
    # Abbreviation patterns
    if re.match(r"^vg\+", s):
        return 6.0
    if re.match(r"^vg\b", s):
        return 5.0
    if re.match(r"^nm\b", s):
        return 7.0
    return -1.0


def format_flags_from_text(desc: str | None) -> dict[str, int]:
    if not desc:
        return {
            "is_colored_vinyl": 0,
            "is_picture_disc": 0,
            "is_promo": 0,
        }
    t = desc.lower()
    return {
        "is_colored_vinyl": int(
            any(x in t for x in ("colored", "coloured", "color vinyl"))
        ),
        "is_picture_disc": int("picture disc" in t or "pic disc" in t),
        "is_promo": int(
            any(x in t for x in ("promo", "promotional", "white label"))
        ),
    }


def parse_json_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except json.JSONDecodeError:
            return []
    return []


def format_medium_flags(
    formats: list[Any],
    format_desc: str | None,
) -> dict[str, float]:
    """LP / 7\" / 12\" / CD hints from structured formats + format_desc."""
    parts: list[str] = []
    for f in formats:
        if not isinstance(f, dict):
            continue
        parts.append(str(f.get("name") or ""))
        parts.extend(str(x) for x in (f.get("descriptions") or []))
    blob = " ".join(parts)
    if format_desc:
        blob = f"{blob} {format_desc}"
    t = blob.lower()
    is_lp = bool(re.search(r"\blp\b", t))
    is_7 = bool(re.search(r"7\s*\"", t)) or bool(re.search(r"\b7\s*inch\b", t))
    is_12 = bool(re.search(r"1\s*2\s*\"", t)) or bool(re.search(r"\b12\s*inch\b", t))
    is_cd = bool(re.search(r"\bcd\b", t)) or ("compact disc" in t)
    return {
        "is_lp": float(is_lp),
        "is_7inch": float(is_7),
        "is_12inch": float(is_12),
        "is_cd": float(is_cd),
    }


def catalog_counts(cat: dict[str, Any]) -> dict[str, float]:
    g = parse_json_list(cat.get("genres_json"))
    if not g and cat.get("genre"):
        g = [cat["genre"]]
    s = parse_json_list(cat.get("styles_json"))
    if not s and cat.get("style"):
        s = [cat["style"]]
    artists = parse_json_list(cat.get("artists_json"))
    labels = parse_json_list(cat.get("labels_json"))
    fmts = parse_json_list(cat.get("formats_json"))
    return {
        "genre_count": float(len(g)),
        "style_count": float(len(s)),
        "artist_count": float(len(artists)),
        "label_count": float(len(labels)),
        "format_count": float(len(fmts)),
    }


def first_artist_id(cat: dict[str, Any]) -> str:
    artists = parse_json_list(cat.get("artists_json"))
    if artists and isinstance(artists[0], dict):
        return str(artists[0].get("id") or "").strip()
    return ""


def first_label_id(cat: dict[str, Any]) -> str:
    labels = parse_json_list(cat.get("labels_json"))
    if labels and isinstance(labels[0], dict):
        return str(labels[0].get("id") or "").strip()
    return ""


def build_model_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    feature_columns: list[str],
) -> np.ndarray:
    """Stack dict rows into a numeric matrix aligned to feature_columns."""
    out = []
    for r in rows:
        out.append([float(r.get(c, 0.0) or 0.0) for c in feature_columns])
    return np.asarray(out, dtype=np.float64)


def row_dict_for_inference(
    release_id: str,
    media_condition: str | None,
    sleeve_condition: str | None,
    stats: dict[str, Any],
    catalog: dict[str, Any] | None,
    *,
    genre_index: float = 0.0,
    country_index: float = 0.0,
    primary_artist_index: float = 0.0,
    primary_label_index: float = 0.0,
    include_marketplace_scalars_in_features: bool = True,
) -> dict[str, float]:
    """Build one numeric feature dict for XGBoost (categoricals encoded externally).

    When ``include_marketplace_scalars_in_features`` is False (residual target), zero out
    ``log_num_for_sale``, ``num_for_sale``, and baseline price fields so X has no same-snapshot
    Discogs price or liquidity. Anchor ``median_price`` is applied after predict.
    """
    media_ord = condition_string_to_ordinal(media_condition)
    sleeve_ord = condition_string_to_ordinal(sleeve_condition)
    # Prefer GET /releases lowest, then marketplace aggregates (median often mirrors lowest).
    lowest = (
        stats.get("release_lowest_price")
        or stats.get("lowest_price")
        or stats.get("median_price")
        or 0.0
    )
    if lowest is None:
        lowest = 0.0
    num_sale = int(stats.get("num_for_sale") or 0)
    if not include_marketplace_scalars_in_features:
        lowest = 0.0
        num_sale = 0
    cat = catalog or {}
    wants = int(cat.get("want_count") or 0)
    haves = int(cat.get("have_count") or 0)
    ratio = float(cat.get("want_have_ratio") or 0.0)
    if ratio == 0.0 and haves > 0:
        ratio = wants / haves
    year = int(cat.get("year") or 0)
    decade = int(cat.get("decade") or (year // 10 * 10 if year else 0))
    fmts = parse_json_list(cat.get("formats_json"))
    fd = cat.get("format_desc")
    format_desc_s = str(fd).strip() if fd else None
    medium = format_medium_flags(fmts, format_desc_s)
    counts = catalog_counts(cat)
    return {
        "release_id": release_id,
        "media_grade": media_ord,
        "sleeve_grade": sleeve_ord,
        "condition_discount": media_ord / 8.0 if media_ord > 0 else 0.0,
        "want_have_ratio": ratio,
        "log_have_count": math.log1p(haves),
        "log_num_for_sale": math.log1p(num_sale),
        "log1p_baseline_median": math.log1p(float(lowest)) if lowest else 0.0,
        "baseline_median": float(lowest),
        "num_for_sale": float(num_sale),
        "decade": float(decade),
        "year": float(year),
        "is_original_pressing": float(cat.get("is_original_pressing") or 0),
        "label_tier": float(cat.get("label_tier") or 0),
        "is_colored_vinyl": float(cat.get("is_colored_vinyl") or 0),
        "is_picture_disc": float(cat.get("is_picture_disc") or 0),
        "is_promo": float(cat.get("is_promo") or 0),
        "country_index": float(country_index),
        "primary_artist_index": float(primary_artist_index),
        "primary_label_index": float(primary_label_index),
        "genre_count": counts["genre_count"],
        "style_count": counts["style_count"],
        "artist_count": counts["artist_count"],
        "label_count": counts["label_count"],
        "format_count": counts["format_count"],
        "is_lp": medium["is_lp"],
        "is_7inch": medium["is_7inch"],
        "is_12inch": medium["is_12inch"],
        "is_cd": medium["is_cd"],
        "genre_index": float(genre_index),
    }


def residual_training_feature_columns() -> list[str]:
    """Feature columns for ``residual_log_median`` target (no price or liquidity in X)."""
    return [
        "media_grade",
        "sleeve_grade",
        "condition_discount",
        "want_have_ratio",
        "log_have_count",
        "decade",
        "year",
        "is_original_pressing",
        "label_tier",
        "is_colored_vinyl",
        "is_picture_disc",
        "is_promo",
        "country_index",
        "primary_artist_index",
        "primary_label_index",
        "genre_count",
        "style_count",
        "artist_count",
        "label_count",
        "format_count",
        "is_lp",
        "is_7inch",
        "is_12inch",
        "is_cd",
        "genre_index",
    ]


def default_feature_columns() -> list[str]:
    """Numeric columns fed to XGBoost (categorical columns pre-encoded to float)."""
    return [
        "media_grade",
        "sleeve_grade",
        "condition_discount",
        "want_have_ratio",
        "log_have_count",
        "log_num_for_sale",
        "log1p_baseline_median",
        "num_for_sale",
        "decade",
        "year",
        "is_original_pressing",
        "label_tier",
        "is_colored_vinyl",
        "is_picture_disc",
        "is_promo",
        "country_index",
        "primary_artist_index",
        "primary_label_index",
        "genre_count",
        "style_count",
        "artist_count",
        "label_count",
        "format_count",
        "is_lp",
        "is_7inch",
        "is_12inch",
        "is_cd",
        "genre_index",
    ]


def apply_condition_log_adjustment(
    log_price: float,
    media_ord: float,
    sleeve_ord: float,
    *,
    alpha: float = -0.06,
    beta: float = -0.04,
    ref_grade: float = 8.0,
) -> float:
    """Shift log-price based on distance from reference mint grade."""
    return (
        log_price
        + alpha * (media_ord - ref_grade)
        + beta * (sleeve_ord - ref_grade)
    )
