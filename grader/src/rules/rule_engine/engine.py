"""
Concrete ``RuleEngine`` class — composes pattern, contradiction, override,
and application mixins.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from grader.src.config_io import load_yaml_mapping
from grader.src.guidelines_identity import guidelines_version_from_mapping

from .application import RuleEngineApplicationMixin
from .contradiction import RuleEngineContradictionMixin
from .overrides import RuleEngineOverrideMixin
from .pattern_compile import RuleEnginePatternMixin

logger = logging.getLogger(__name__)


class RuleEngine(
    RuleEnginePatternMixin,
    RuleEngineContradictionMixin,
    RuleEngineOverrideMixin,
    RuleEngineApplicationMixin,
):
    """
    Post-processing rule engine grounding model predictions in the
    official Discogs grading rubric and curated signal lists.

    All signal lists, thresholds, and grade ownership are loaded
    from grading_guidelines.yaml — nothing is hardcoded.

    Signal matching uses compiled word-boundary regex patterns for
    accuracy. Partial matches (e.g. "vg+" inside "vg++") are prevented
    by the boundary anchors.

    Args:
        guidelines_path: path to grading_guidelines.yaml
        allow_excellent_soft_override: when False (default),
            :meth:`check_soft_override` never returns ``Excellent``, and
            :meth:`apply` remaps any remaining ``Excellent`` prediction
            (model or rules) to ``Near Mint``. Set True to keep Excellent as a
            live grade (see ``rules.allow_excellent_soft_override`` in
            grader.yaml).
    """

    guidelines: dict
    grade_defs: dict
    _patterns: dict[str, dict[str, list[re.Pattern]]]
    _hard_sources: dict[str, dict[str, list[str]]]
    _hard_patterns: dict[str, dict[str, list[re.Pattern]]]
    _last_hard_tier: Optional[str]
    _last_hard_signal: Optional[str]
    _contradiction_patterns: list[tuple[re.Pattern, re.Pattern]]
    _nm_sleeve_downgrade_patterns: list[re.Pattern]
    _nm_small_split_patterns: list[re.Pattern]
    _nm_split_other_defect_patterns: list[re.Pattern]
    _allow_excellent_soft_override: bool

    def __init__(
        self,
        guidelines_path: str,
        *,
        allow_excellent_soft_override: bool = False,
    ) -> None:
        self.guidelines = load_yaml_mapping(guidelines_path)
        self.grade_defs = self.guidelines["grades"]
        self._allow_excellent_soft_override = allow_excellent_soft_override

        # Pre-compile all signal patterns for performance
        # Structure: {grade: {signal_type: [compiled_pattern, ...]}}
        self._patterns: dict[str, dict[str, list[re.Pattern]]] = {}
        # Populated by _compile_patterns — hard-signal sources kept
        # alongside compiled patterns for logging and diagnostics.
        self._hard_sources: dict[str, dict[str, list[str]]] = {}
        self._hard_patterns: dict[str, dict[str, list[re.Pattern]]] = {}
        # Last hard-override decision cache used by ``apply`` to surface
        # which tier (strict vs cosignal) triggered an override in metadata.
        self._last_hard_tier: Optional[str] = None
        self._last_hard_signal: Optional[str] = None
        self._compile_patterns()

        # Pre-compile contradiction pairs
        self._contradiction_patterns: list[tuple[re.Pattern, re.Pattern]] = (
            self._compile_contradictions()
        )

        self._nm_sleeve_downgrade_patterns = [
            self._compile_signal("seam split"),
            self._compile_signal("spine split"),
        ]
        self._nm_small_split_patterns = [
            self._compile_signal("small seam split"),
            self._compile_signal("small top seam split"),
            self._compile_signal("small seam split along"),
            self._compile_signal("small seam split on top"),
            self._compile_signal("small spine split"),
            self._compile_signal("small top spine split"),
        ]
        self._nm_split_other_defect_patterns = [
            self._compile_signal(s)
            for s in [
                "scratch",
                "scratches",
                "scuff",
                "scuffs",
                "wear",
                "wrinkle",
                "wrinkles",
                "ring wear",
                "ringwear",
                "stain",
                "stains",
                "foxing",
                "mold",
                "water damage",
                "corner",
                "bump",
                "cutout",
                "cut out",
                "tear",
                "tears",
                "fade",
                "writing",
                "sticker",
                "gouge",
                "gouges",
            ]
        ]

        _gv = guidelines_version_from_mapping(self.guidelines)
        logger.info(
            "RuleEngine initialized — guidelines_version=%s | %d grades | "
            "%d contradiction pairs",
            _gv,
            len(self.grade_defs),
            len(self._contradiction_patterns),
        )
