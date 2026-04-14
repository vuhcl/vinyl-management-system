"""Build training dollar targets from marketplace_stats (VinylIQ labels)."""
from __future__ import annotations

import json
from typing import Any


def _positive(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def parse_price_suggestion_value(
    raw_json: str | None,
    grade_key: str,
) -> float | None:
    """
    Read USD (or numeric) ``value`` from marketplace price_suggestions JSON.

    Keys look like ``"Near Mint (NM or M-)"`` with a ``value`` field.
    """
    if raw_json is None or not str(raw_json).strip():
        return None
    try:
        d = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not d:
        return None
    entry = d.get(grade_key)
    if not isinstance(entry, dict):
        return None
    return _positive(entry.get("value"))


def dollar_target_and_residual_anchor_from_marketplace_row(
    row: dict[str, Any],
    tl: dict[str, Any],
) -> tuple[float | None, float | None]:
    """
    Compute training dollar target ``y`` and residual anchor ``m`` for
    **non-sale-floor** label modes (tests, one-off analysis).

    VinylIQ **training** uses ``sale_floor_blend`` via ``load_training_frame``
    and ``sale_floor_blend_bundle``; do not use this helper for that path.
    """
    mode = str(tl.get("mode", "median")).strip().lower()
    grade = str(tl.get("price_suggestion_grade") or "Near Mint (NM or M-)").strip()
    fallback_ps = bool(tl.get("price_suggestion_fallback_lowest", True))

    anchor = (
        _positive(row.get("release_lowest_price"))
        or _positive(row.get("lowest_price"))
        or _positive(row.get("median_price"))
    )

    if mode in ("price_suggestion", "price_suggestion_nm"):
        psj = row.get("price_suggestions_json")
        y = parse_price_suggestion_value(psj, grade)
        if y is None and fallback_ps:
            y = (
                _positive(row.get("release_lowest_price"))
                or _positive(row.get("lowest_price"))
                or _positive(row.get("median_price"))
            )
        if y is None:
            return None, None
        m = anchor if anchor is not None else y
        return float(y), float(m)

    if mode in ("price_suggestion_strict",):
        psj = row.get("price_suggestions_json")
        y = parse_price_suggestion_value(psj, grade)
        if y is None:
            return None, None
        m = anchor if anchor is not None else y
        return float(y), float(m)

    if mode in ("release_lowest", "lowest"):
        y = (
            _positive(row.get("release_lowest_price"))
            or _positive(row.get("lowest_price"))
            or _positive(row.get("median_price"))
        )
        if y is None:
            return None, None
        yy = float(y)
        return yy, yy

    if mode in ("sale_floor_blend", "sale_floor"):
        raise ValueError(
            "training_label.mode sale_floor_blend is built in load_training_frame "
            "with sale_history.sqlite; do not call "
            "dollar_target_and_residual_anchor_from_marketplace_row alone."
        )

    raise ValueError(
        f"Unknown or retired training_label.mode: {mode!r}. "
        "Use sale_floor_blend or sale_floor for training, or "
        "price_suggestion or release_lowest for this helper."
    )


def training_label_config_from_vinyliq(
    v: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read ``vinyliq.training_label`` from config with defaults."""
    raw = (v or {}).get("training_label") or {}
    if not isinstance(raw, dict):
        raw = {}
    mode = str(raw.get("mode", "sale_floor_blend")).strip().lower()
    psg = raw.get("price_suggestion_grade", "Near Mint (NM or M-)")
    ps_grade = str(psg).strip() if psg else "Near Mint (NM or M-)"
    ps_fb = raw.get("price_suggestion_fallback_lowest", True)
    if isinstance(ps_fb, str):
        ps_fb = ps_fb.strip().lower() in ("1", "true", "yes", "on")
    sfb = raw.get("sale_floor_blend")
    if sfb is not None and not isinstance(sfb, dict):
        sfb = {}
    return {
        "mode": mode,
        "price_suggestion_grade": ps_grade,
        "price_suggestion_fallback_lowest": bool(ps_fb),
        "sale_floor_blend": dict(sfb) if isinstance(sfb, dict) else {},
    }
