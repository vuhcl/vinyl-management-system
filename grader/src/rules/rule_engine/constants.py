"""Shared constants for the rule engine (YAML key lists, grade sets, targets)."""

# Optional YAML keys — compiled when present (see PatternCompileMixin._compile_patterns).
_SOFT_PATTERN_KEYS = (
    "supporting_signals",
    "forbidden_signals",
    "supporting_signals_sleeve",
    "supporting_signals_media",
    "forbidden_signals_sleeve",
    "forbidden_signals_media",
    # Exception phrases: matched substrings are stripped from text before the
    # forbidden check, allowing them to neutralise specific forbidden signals.
    # E.g. "sticker residue" as an exception cancels the "sticker" forbidden so
    # a VG+ override is not blocked just because residue was left behind.
    "forbidden_exceptions_sleeve",
    "forbidden_exceptions_media",
)

# Hard signal keys — split into strict (single match fires) and cosignal
# (requires corroboration from another distinct signal in the same grade).
# Per-target variants override the untargeted ones when present; all fall
# back to the legacy ``hard_signals`` list for back-compat.
_HARD_PATTERN_KEYS = (
    "hard_signals",
    "hard_signals_strict",
    "hard_signals_cosignal",
    "hard_signals_strict_sleeve",
    "hard_signals_strict_media",
    "hard_signals_cosignal_sleeve",
    "hard_signals_cosignal_media",
)

# Grades owned exclusively by the rule engine (hard + soft overrides apply)
HARD_OWNED_GRADES = {"Poor", "Generic"}

# Grades owned exclusively by the model — rule engine never overrides these.
# Mint hard-override fires correctly for a subset (sealed listings) but causes
# large-scale harm across the full dataset; model accuracy is higher overall.
MODEL_ONLY_GRADES = {"Mint"}

# Targets
SLEEVE = "sleeve"
MEDIA = "media"

# Hard-owned grades: evaluate Poor before Generic on sleeve so jacket
# catastrophe wins over housing-type keywords (see package docstring).
_HARD_OWNED_ORDER_DEFAULT: tuple[str, ...] = ("Poor", "Generic")
