"""Build training regression targets from marketplace_stats scalars."""
from __future__ import annotations

import json
import math
from typing import Any


def _positive(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def _non_negative_count(v: Any) -> int:
    if v is None:
        return 0
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def synthesize_training_price(
    median_price: Any,
    lowest_price: Any,
    *,
    mode: str = "median",
    blend_median_weight: float = 0.7,
    spread_lowest_floor_ratio: float | None = None,
    spread_min_median_weight: float | None = None,
    num_for_sale: Any | None = None,
    spread_num_for_sale_reference: float | None = None,
) -> float | None:
    """
    Combine Discogs aggregate listing prices into one dollar target for ``log1p(y)``.

    - ``median``: median only (legacy).
    - ``blend``: ``w * median + (1-w) * lowest``; missing lowest → median.
      ``w`` is ``blend_median_weight`` clamped to [0, 1].
    - ``geometric_mean``: ``sqrt(median * lowest)``; missing lowest → median.
    - ``spread_signal``: let ``c = min(1, lowest/median)`` measure how tight the
      book is (``lowest`` tracks ``median``). Use
      ``y = c * median + (1 - c) * lowest``.
      **Tight** (``c → 1``): trust the aggregate → **``y → median``**.
      **Wide spread** (``c → 0``): ``median`` may sit above a thin floor → **``y``
      moves toward ``lowest``** instead of always hugging ``median``.
      Missing lowest → median only. ``lowest > median`` → ``c`` capped at 1 → ``y = median``.

      **Robustness** (optional, recommended for training): Discogs ``lowest`` is often a
      damaged or incomplete listing. With ``spread_lowest_floor_ratio`` in ``(0, 1]``,
      use ``lowest_eff = max(lowest, ratio * median)`` everywhere above instead of raw
      ``lowest``. With ``spread_min_median_weight`` in ``[0, 1]``, enforce
      ``c = max(that, min_median_weight)`` so some weight always stays on the median.

      **Listing depth × spread width** (optional, ``spread_signal`` only): when
      ``spread_num_for_sale_reference > 0``, let ``y_s`` be the spread target above,
      ``r = min(1, lowest_eff / median)``, ``gap = 1 - r`` (wide book → larger gap),
      and ``n = num_for_sale`` (missing or non-positive → ``0``). With
      ``t = min(1, n / reference)``, use ``pull = gap * t`` and
      ``y = (1 - pull) * y_s + pull * median``. **Many listings + wide spread** → median
      is a stable aggregate while the floor may be an outlier → stronger pull to median.
      **Few listings + wide spread** → little pull → keep ``y_s`` (floor signal). **Tight
      spread** (``gap ≈ 0``) → no pull regardless of ``n``.

    Non-positive median → None (SQL also filters ``median_price > 0``).
    """
    m = _positive(median_price)
    if m is None:
        return None
    mode = str(mode or "median").strip().lower()
    if mode == "median":
        return m
    lo = _positive(lowest_price)
    if mode == "blend":
        w = float(blend_median_weight)
        w = max(0.0, min(1.0, w))
        if lo is None:
            return m
        return w * m + (1.0 - w) * lo
    if mode in ("geometric_mean", "geometric", "gmean"):
        if lo is None:
            return m
        return float(math.sqrt(m * lo))
    if mode in ("spread_signal", "spread"):
        if lo is None:
            return m
        lo_eff = lo
        fr = spread_lowest_floor_ratio
        if fr is not None and fr > 0:
            fr = min(1.0, float(fr))
            lo_eff = max(lo_eff, fr * m)
        raw_c = min(1.0, lo_eff / m)
        c = raw_c
        mw = spread_min_median_weight
        if mw is not None:
            mw = max(0.0, min(1.0, float(mw)))
            c = max(mw, min(1.0, raw_c))
        y_spread = c * m + (1.0 - c) * lo_eff
        ref = spread_num_for_sale_reference
        if ref is None or ref <= 0:
            return y_spread
        n_list = _non_negative_count(num_for_sale)
        listing_median_trust = min(1.0, float(n_list) / float(ref))
        r = min(1.0, lo_eff / m)
        spread_gap = max(0.0, min(1.0, 1.0 - r))
        pull = spread_gap * listing_median_trust
        return (1.0 - pull) * y_spread + pull * m
    raise ValueError(f"Unknown training_label.mode: {mode!r}")


def parse_price_suggestion_value(
    raw_json: str | None,
    grade_key: str,
) -> float | None:
    """
    Read USD (or numeric) ``value`` from ``GET /marketplace/price_suggestions`` JSON.

    Keys look like ``"Near Mint (NM or M-)"`` → ``{"currency": "USD", "value": …}``.
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


def _spread_compatible_modes() -> tuple[str, ...]:
    return ("median", "blend", "geometric_mean", "geometric", "gmean", "spread_signal", "spread")


def dollar_target_and_residual_anchor_from_marketplace_row(
    row: dict[str, Any],
    tl: dict[str, Any],
) -> tuple[float | None, float | None]:
    """
    Compute training dollar target ``y`` and residual anchor ``m`` (for
    ``log1p(y) - log1p(m)``). Anchor prefers ``release_lowest_price`` from
    ``GET /releases``, then marketplace stats lowest, then stored median.
    """
    mode = str(tl.get("mode", "median")).strip().lower()
    grade = str(tl.get("price_suggestion_grade") or "Near Mint (NM or M-)").strip()
    fallback_ps = bool(tl.get("price_suggestion_fallback_lowest", True))

    anchor = (
        _positive(row.get("release_lowest_price"))
        or _positive(row.get("lowest_price"))
        or _positive(row.get("median_price"))
    )

    median_spread = (
        _positive(row.get("release_lowest_price"))
        or _positive(row.get("median_price"))
        or _positive(row.get("lowest_price"))
    )
    lowest_spread = (
        _positive(row.get("lowest_price"))
        or _positive(row.get("release_lowest_price"))
    )

    if mode in ("price_suggestion", "price_suggestion_nm"):
        y = parse_price_suggestion_value(row.get("price_suggestions_json"), grade)
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
        y = parse_price_suggestion_value(row.get("price_suggestions_json"), grade)
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
            "training_label.mode 'sale_floor_blend' is built in load_training_frame "
            "with sale_history.sqlite (sold nowcast + listing floor); "
            "do not call dollar_target_and_residual_anchor_from_marketplace_row alone."
        )

    if mode in _spread_compatible_modes():
        y = synthesize_training_price(
            median_spread,
            lowest_spread,
            mode=mode,
            blend_median_weight=float(tl.get("blend_median_weight", 0.7)),
            spread_lowest_floor_ratio=tl.get("spread_lowest_floor_ratio"),
            spread_min_median_weight=tl.get("spread_min_median_weight"),
            num_for_sale=row.get("num_for_sale"),
            spread_num_for_sale_reference=tl.get("spread_num_for_sale_reference"),
        )
        if y is None:
            return None, None
        m = anchor if anchor is not None else float(y)
        return float(y), float(m)

    raise ValueError(f"Unknown training_label.mode: {mode!r}")


def _parse_spread_floor_ratio(raw: dict[str, Any], mode: str) -> float | None:
    key = "spread_lowest_floor_ratio"
    spread_modes = ("spread_signal", "spread")
    default = 0.35 if mode in spread_modes else None
    if key not in raw:
        return default
    v = raw.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x <= 0:
        return None
    return min(1.0, x)


def _parse_spread_min_median_weight(raw: dict[str, Any], mode: str) -> float | None:
    key = "spread_min_median_weight"
    spread_modes = ("spread_signal", "spread")
    default = 0.25 if mode in spread_modes else None
    if key not in raw:
        return default
    v = raw.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x <= 0:
        return None
    return min(1.0, x)


def _parse_spread_num_for_sale_reference(raw: dict[str, Any], mode: str) -> float | None:
    key = "spread_num_for_sale_reference"
    spread_modes = ("spread_signal", "spread")
    default = 20.0 if mode in spread_modes else None
    if key not in raw:
        return default
    v = raw.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x <= 0:
        return None
    return x


def training_label_config_from_vinyliq(
    v: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read ``vinyliq.training_label`` from config with defaults."""
    raw = (v or {}).get("training_label") or {}
    if not isinstance(raw, dict):
        raw = {}
    mode = str(raw.get("mode", "median")).strip().lower()
    w = raw.get("blend_median_weight", 0.7)
    try:
        blend_w = float(w)
    except (TypeError, ValueError):
        blend_w = 0.7
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
        "blend_median_weight": blend_w,
        "spread_lowest_floor_ratio": _parse_spread_floor_ratio(raw, mode),
        "spread_min_median_weight": _parse_spread_min_median_weight(raw, mode),
        "spread_num_for_sale_reference": _parse_spread_num_for_sale_reference(raw, mode),
        "price_suggestion_grade": ps_grade,
        "price_suggestion_fallback_lowest": bool(ps_fb),
        "sale_floor_blend": dict(sfb) if isinstance(sfb, dict) else {},
    }
