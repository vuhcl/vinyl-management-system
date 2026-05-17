"""
VinylIQ feature assembly: ordinal conditions + catalog + marketplace stats.

Used by training and inference. Genre is passed through as string; training
fits a target encoder saved in artifacts.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar

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

MAX_PRICE_USD: float = 100_000.0
MAX_LOG_PRICE: float = math.log1p(MAX_PRICE_USD)


def clamp_ordinals_for_inference(media_ord: float, sleeve_ord: float) -> tuple[float, float]:
    """Clamp condition ordinals to [1, 8] before user-facing condition adjustment."""
    return (
        max(1.0, min(8.0, float(media_ord))),
        max(1.0, min(8.0, float(sleeve_ord))),
    )


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


def _description_is_repress_token(raw: str) -> bool:
    """True if Discogs format description is the **Repress** pressing tag."""
    s = str(raw).strip().lower()
    return s == "repress"


def is_original_pressing_from_formats_list(
    formats: list[Any] | None,
) -> int:
    """
    Plan §1a: ``1`` when no format carries the **Repress** description; ``0`` if any
    does. **Empty / missing formats → ``1``** (not labeled as repress in format data).
    """
    if not formats:
        return 1
    for f in formats:
        if not isinstance(f, dict):
            continue
        for d in f.get("descriptions") or []:
            if _description_is_repress_token(str(d)):
                return 0
    return 1


def is_original_pressing_from_formats_json(formats_json: str | None) -> int:
    """§1a from stored ``formats_json`` (same semantics as ``is_original_pressing_from_formats_list``)."""
    fmts = parse_json_list(formats_json) if formats_json else []
    return is_original_pressing_from_formats_list(fmts)


def is_original_pressing_from_format_desc(format_desc: str | None) -> int:
    """
    Fallback when only a flat ``format_desc`` string exists (e.g. CSV ingest).
    Treat whole-token **repress** as repress-labeled (conservative substring on word bounds).
    """
    if not format_desc:
        return 1
    t = format_desc.lower()
    if re.search(r"\brepress\b", t):
        return 0
    return 1


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


def _format_qty_sum(formats: list[Any]) -> int:
    """Sum Discogs ``formats[].qty`` when present (multi-disc / box hints)."""
    total = 0
    for f in formats:
        if not isinstance(f, dict):
            continue
        q = f.get("qty")
        if q is None or not str(q).strip():
            continue
        try:
            qi = int(float(str(q).strip()))
        except (TypeError, ValueError):
            continue
        if qi > 0:
            total += qi
    return total


def format_medium_flags(
    formats: list[Any],
    format_desc: str | None,
) -> dict[str, float]:
    """LP / 7\" / 10\" / 12\" / CD + box set / multi-disc + ordinal family."""
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
    is_10 = bool(re.search(r"10\s*\"", t)) or bool(re.search(r"\b10\s*inch\b", t))
    is_cd = bool(re.search(r"\bcd\b", t)) or ("compact disc" in t)
    is_box = "box set" in t
    qty_sum = _format_qty_sum(formats)
    multi_phrase = (
        bool(re.search(r"\b[2-9]\s*x\s*vinyl\b", t))
        or "double vinyl" in t
        or "triple vinyl" in t
        or ("quad" in t and "vinyl" in t)
        or (qty_sum >= 2)
    )
    is_multi = float(bool(multi_phrase and not is_box))
    is_box_f = float(is_box)

    # Ordinal: cheap singles vs LP/box (trees also get raw bits).
    fmt_family = 0.0
    if is_box or multi_phrase:
        fmt_family = 5.0
    elif is_7:
        fmt_family = 2.0
    elif is_10:
        fmt_family = 3.0
    elif is_12:
        fmt_family = 3.0
    elif is_lp:
        fmt_family = 4.0
    elif is_cd:
        fmt_family = 1.0

    return {
        "is_lp": float(is_lp),
        "is_7inch": float(is_7),
        "is_10inch": float(is_10),
        "is_12inch": float(is_12),
        "is_cd": float(is_cd),
        "is_box_set": is_box_f,
        "is_multi_disc": is_multi,
        "format_family": float(fmt_family),
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


def _int_nonneg_or_zero(v: Any) -> int:
    if v is None:
        return 0
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _positive_listing_scalar(v: Any) -> bool:
    if v is None:
        return False
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def _community_counts_from_stats_or_catalog(
    stats: dict[str, Any],
    cat: dict[str, Any],
) -> tuple[int, int]:
    """Plan §1b: community counts from marketplace row only (``cat`` unused, kept for API)."""
    _ = cat
    cw = _int_nonneg_or_zero(stats.get("community_want"))
    ch = _int_nonneg_or_zero(stats.get("community_have"))
    return cw, ch


def _marketplace_depth_feature_block(
    stats: dict[str, Any],
    cat: dict[str, Any],
) -> dict[str, float]:
    """Non-dollar marketplace signals (plan §4d): community, listing depth, blocked."""
    cw, ch = _community_counts_from_stats_or_catalog(stats, cat)
    log1p_w = math.log1p(float(cw))
    log1p_h = math.log1p(float(ch))
    if cw + ch > 0:
        want_share = float(cw) / float(cw + ch)
        has_community = 1.0
    else:
        want_share = -1.0
        has_community = 0.0

    num_sale = _int_nonneg_or_zero(stats.get("num_for_sale"))
    log_num_for_sale = math.log1p(float(num_sale))

    raw_rnfs = stats.get("release_num_for_sale")
    if raw_rnfs is None:
        log_release_num_for_sale = 0.0
        has_release_num_for_sale = 0.0
    else:
        rn = _int_nonneg_or_zero(raw_rnfs)
        log_release_num_for_sale = math.log1p(float(rn))
        has_release_num_for_sale = 1.0

    if stats.get("num_for_sale") is not None and stats.get("release_num_for_sale") is not None:
        log_nfs_delta = math.log1p(float(num_sale)) - math.log1p(
            float(_int_nonneg_or_zero(stats.get("release_num_for_sale"))),
        )
        has_nfs_delta = 1.0
    else:
        log_nfs_delta = 0.0
        has_nfs_delta = 0.0

    bl = stats.get("blocked_from_sale")
    if bl is not None:
        try:
            blocked = 1.0 if int(bl) else 0.0
            has_blocked = 1.0
        except (TypeError, ValueError):
            blocked = 0.0
            has_blocked = 0.0
    else:
        blocked = 0.0
        has_blocked = 0.0

    return {
        "log1p_community_want": log1p_w,
        "log1p_community_have": log1p_h,
        "want_share": want_share,
        "has_community": has_community,
        "log_num_for_sale": log_num_for_sale,
        "num_for_sale": float(num_sale),
        "log_release_num_for_sale": log_release_num_for_sale,
        "has_release_num_for_sale": has_release_num_for_sale,
        "log_nfs_delta": log_nfs_delta,
        "has_nfs_delta": has_nfs_delta,
        "blocked_from_sale": blocked,
        "has_blocked_from_sale": has_blocked,
    }


def _catalog_tail(
    cat: dict[str, Any],
    *,
    genre_index: float,
    country_index: float,
    primary_artist_index: float,
    primary_label_index: float,
) -> dict[str, float]:
    year = int(cat.get("year") or 0)
    decade = int(cat.get("decade") or (year // 10 * 10 if year else 0))
    fmts = parse_json_list(cat.get("formats_json"))
    fd = cat.get("format_desc")
    format_desc_s = str(fd).strip() if fd else None
    medium = format_medium_flags(fmts, format_desc_s)
    counts = catalog_counts(cat)
    return {
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
        "is_10inch": medium["is_10inch"],
        "is_12inch": medium["is_12inch"],
        "is_cd": medium["is_cd"],
        "is_box_set": medium["is_box_set"],
        "is_multi_disc": medium["is_multi_disc"],
        "format_family": medium["format_family"],
        "genre_index": float(genre_index),
    }


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
    cold_start_flags: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build one numeric feature dict for XGBoost (categoricals encoded externally).

    When ``include_marketplace_scalars_in_features`` is False (``residual_log_median``),
    same-snapshot **listing-dollar** scalars (``log1p_baseline_median``, ``baseline_median``)
    are zeroed so the tree does not see the cheapest listing / aggregate median as price
    features. **Non-dollar** marketplace depth (``log_num_for_sale``, community counts,
    ``blocked_from_sale``, etc.) still comes from ``stats`` per plan §4d. The residual anchor
    is applied after predict (pyfunc: ``discogs_median_price``).
    """
    media_ord = condition_string_to_ordinal(media_condition)
    sleeve_ord = condition_string_to_ordinal(sleeve_condition)
    lowest = stats.get("release_lowest_price") or 0.0
    if lowest is None:
        lowest = 0.0
    if not include_marketplace_scalars_in_features:
        lowest = 0.0

    cat = catalog or {}
    depth = _marketplace_depth_feature_block(stats, cat)
    lo_pos = bool(_positive_listing_scalar(stats.get("release_lowest_price")))
    if cold_start_flags is not None:
        has_sh = float(cold_start_flags.get("has_sale_history", 0.0) or 0.0)
        s_imp = float(cold_start_flags.get("s_imputed", 0.0) or 0.0)
        has_lf = float(cold_start_flags.get("has_listing_floor", 0.0) or 0.0)
    else:
        has_sh = 0.0
        s_imp = 0.0
        has_lf = 1.0 if lo_pos else 0.0
    depth["has_sale_history"] = has_sh
    depth["s_imputed"] = s_imp
    depth["has_listing_floor"] = has_lf
    tail = _catalog_tail(
        cat,
        genre_index=genre_index,
        country_index=country_index,
        primary_artist_index=primary_artist_index,
        primary_label_index=primary_label_index,
    )
    out: dict[str, float] = {
        "release_id": release_id,
        "media_grade": media_ord,
        "sleeve_grade": sleeve_ord,
        "condition_discount": media_ord / 8.0 if media_ord > 0 else 0.0,
        **depth,
        "log1p_baseline_median": math.log1p(float(lowest)) if lowest else 0.0,
        "baseline_median": float(lowest),
        **tail,
    }
    if not include_marketplace_scalars_in_features:
        out["log1p_baseline_median"] = 0.0
        out["baseline_median"] = 0.0
    return out


@dataclass(frozen=True)
class VinylIQFeatureSchema:
    """
    Single source of truth for model-matrix column order.

    ``default`` adds listing-dollar log features between marketplace depth and catalog tail;
    ``residual_log_median`` training omits those so trees do not see listing medians as X.
    """

    CONDITION_HEAD: ClassVar[tuple[str, ...]] = (
        "media_grade",
        "sleeve_grade",
        "condition_discount",
    )
    COLD_START: ClassVar[tuple[str, ...]] = (
        "has_sale_history",
        "s_imputed",
        "has_listing_floor",
    )
    MARKETPLACE_DEPTH_BODY: ClassVar[tuple[str, ...]] = (
        "log1p_community_want",
        "log1p_community_have",
        "want_share",
        "has_community",
        "log_num_for_sale",
        "num_for_sale",
        "log_release_num_for_sale",
        "has_release_num_for_sale",
        "log_nfs_delta",
        "has_nfs_delta",
        "blocked_from_sale",
        "has_blocked_from_sale",
    )
    LISTING_DOLLAR: ClassVar[tuple[str, ...]] = ("log1p_baseline_median",)
    CATALOG_TAIL: ClassVar[tuple[str, ...]] = (
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
        "is_10inch",
        "is_12inch",
        "is_cd",
        "is_box_set",
        "is_multi_disc",
        "format_family",
        "genre_index",
    )

    @classmethod
    def marketplace_depth_columns(cls) -> tuple[str, ...]:
        return cls.MARKETPLACE_DEPTH_BODY + cls.COLD_START

    @classmethod
    def residual_training_columns(cls) -> list[str]:
        return list(cls.CONDITION_HEAD + cls.marketplace_depth_columns() + cls.CATALOG_TAIL)

    @classmethod
    def default_training_columns(cls) -> list[str]:
        return list(
            cls.CONDITION_HEAD
            + cls.marketplace_depth_columns()
            + cls.LISTING_DOLLAR
            + cls.CATALOG_TAIL
        )


def _cold_start_feature_column_names() -> list[str]:
    """Plan §4d / §7.1b: training-time missingness for sale history and listing floor."""
    return list(VinylIQFeatureSchema.COLD_START)


def _marketplace_depth_column_names() -> list[str]:
    return list(VinylIQFeatureSchema.marketplace_depth_columns())


def residual_training_feature_columns() -> list[str]:
    """Feature columns for ``residual_log_median`` target (no listing-dollar columns in X)."""
    return VinylIQFeatureSchema.residual_training_columns()


def default_feature_columns() -> list[str]:
    """Numeric columns fed to XGBoost (categorical columns pre-encoded to float)."""
    return VinylIQFeatureSchema.default_training_columns()


def apply_condition_log_adjustment(
    log_price: float,
    media_ord: float,
    sleeve_ord: float,
    *,
    alpha: float = 0.06,
    beta: float = 0.04,
    ref_grade: float = 8.0,
) -> float:
    """Shift log-price based on distance from reference mint grade."""
    return (
        log_price
        + alpha * (media_ord - ref_grade)
        + beta * (sleeve_ord - ref_grade)
    )


@dataclass(frozen=True)
class GradeDeltaScaleParams:
    """
    Optional multipliers for condition ``alpha`` / ``beta`` by anchor price and year.

    When ``price_gamma`` and ``age_k`` are both zero, ``grade_scale_from_params`` is 1.0
    (legacy constant adjustment).
    """

    price_ref_usd: float = 50.0
    price_gamma: float = 0.0
    price_scale_min: float = 0.25
    price_scale_max: float = 4.0
    age_k: float = 0.0
    age_center_year: float = 2000.0

    @staticmethod
    def from_mapping(raw: dict[str, Any] | None) -> GradeDeltaScaleParams | None:
        if not raw or not isinstance(raw, dict):
            return None
        def _f(key: str, default: float) -> float:
            v = raw.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        return GradeDeltaScaleParams(
            price_ref_usd=_f("price_ref_usd", 50.0),
            price_gamma=_f("price_gamma", 0.0),
            price_scale_min=_f("price_scale_min", 0.25),
            price_scale_max=_f("price_scale_max", 4.0),
            age_k=_f("age_k", 0.0),
            age_center_year=_f("age_center_year", 2000.0),
        )


def grade_delta_scale_params_from_cond(cond: dict[str, Any] | None) -> GradeDeltaScaleParams | None:
    """Read nested ``grade_delta_scale`` from merged condition params (YAML / JSON)."""
    if not cond:
        return None
    raw = cond.get("grade_delta_scale")
    return GradeDeltaScaleParams.from_mapping(raw) if raw is not None else None


def scale_price_from_anchor(anchor_usd: float, p: GradeDeltaScaleParams) -> float:
    if p.price_gamma == 0.0:
        return 1.0
    a = max(float(anchor_usd), 1e-6)
    den = math.log1p(max(float(p.price_ref_usd), 1e-6))
    if den <= 1e-12:
        return 1.0
    raw = (math.log1p(a) / den) ** float(p.price_gamma)
    lo, hi = float(p.price_scale_min), float(p.price_scale_max)
    return max(lo, min(hi, float(raw)))


def scale_age_from_year(year: float | None, p: GradeDeltaScaleParams) -> float:
    if p.age_k == 0.0 or year is None:
        return 1.0
    try:
        y = float(year)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(y):
        return 1.0
    raw = 1.0 + float(p.age_k) * (float(p.age_center_year) - y) / 50.0
    return max(0.25, min(4.0, float(raw)))


def grade_scale_from_params(
    anchor_usd: float,
    release_year: float | None,
    p: GradeDeltaScaleParams | None,
) -> float:
    if p is None:
        return 1.0
    return scale_price_from_anchor(anchor_usd, p) * scale_age_from_year(release_year, p)


def scaled_condition_log_adjustment(
    log_price: float,
    media_ord: float,
    sleeve_ord: float,
    *,
    base_alpha: float,
    base_beta: float,
    ref_grade: float,
    anchor_usd: float | None = None,
    release_year: float | None = None,
    scale_params: GradeDeltaScaleParams | None = None,
) -> float:
    """
    Same law as ``apply_condition_log_adjustment`` with ``alpha_eff = base_alpha * g``,
    ``beta_eff = base_beta * g``, ``g = grade_scale_from_params(anchor, year, scale)``.

    When ``scale_params`` is None or both ``price_gamma`` and ``age_k`` are zero,
    delegates to ``apply_condition_log_adjustment`` for bitwise parity with legacy.
    """
    if scale_params is None or (
        scale_params.price_gamma == 0.0 and scale_params.age_k == 0.0
    ):
        return apply_condition_log_adjustment(
            log_price,
            media_ord,
            sleeve_ord,
            alpha=base_alpha,
            beta=base_beta,
            ref_grade=ref_grade,
        )
    anchor = (
        float(anchor_usd)
        if anchor_usd is not None and float(anchor_usd) > 0
        else 1.0
    )
    g = grade_scale_from_params(anchor, release_year, scale_params)
    ae = float(base_alpha) * g
    be = float(base_beta) * g
    return log_price + ae * (media_ord - ref_grade) + be * (sleeve_ord - ref_grade)


def log1p_nm_equivalent_from_sale_usd(
    sale_usd: float,
    media_ord: float,
    sleeve_ord: float,
    nm_media_ord: float,
    nm_sleeve_ord: float,
    *,
    base_alpha: float,
    base_beta: float,
    anchor_usd: float,
    release_year: float | None,
    scale_params: GradeDeltaScaleParams | None,
) -> float:
    """``log1p`` dollar at NM reference ordinals (sale-history uplift), shared law as pyfunc."""
    log_obs = math.log1p(max(float(sale_usd), 0.0))
    g = grade_scale_from_params(max(float(anchor_usd), 1e-6), release_year, scale_params)
    return log_obs + float(base_alpha) * g * (float(nm_media_ord) - float(media_ord)) + float(
        base_beta
    ) * g * (float(nm_sleeve_ord) - float(sleeve_ord))
