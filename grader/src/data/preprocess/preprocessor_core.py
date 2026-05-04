"""Preprocessor orchestration for the vinyl condition grader."""

from __future__ import annotations

import copy
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import mlflow
import yaml

from grader.src.mlflow_tracking import (
    mlflow_pipeline_step_run_ctx,
)
from sklearn.model_selection import StratifiedShuffleSplit

from .listing_promo import (
    load_promo_noise_patterns,
    protected_terms_from_grades,
    strip_listing_promo_noise,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Text preprocessing pipeline for vinyl condition grader.

    Config keys read from grader.yaml:
        preprocessing.lowercase
        preprocessing.normalize_whitespace
        preprocessing.strip_stray_numeric_tokens
        preprocessing.promo_noise_patterns
        preprocessing.abbreviation_map
        preprocessing.min_text_length_discogs
        data.splits.train / val / test
        data.splits.random_seed
        paths.processed
        paths.splits
        mlflow.tracking_uri
        mlflow.experiment_name

    Config keys read from grading_guidelines.yaml:
        grades.Mint.hard_signals          — for unverified media detection
        grades.Generic.hard_signals*      — for Generic sleeve detection
            (aggregated across legacy ``hard_signals`` plus the
            strict/cosignal variants introduced in §13/§13b; see
            :func:`_collect_hard_signals`)
        grades[*].*signal* lists
            — protected terms for cleaning / gating (all keys whose names
              contain ``signal``)
    """

    def __init__(
        self,
        config_path: str,
        guidelines_path: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = self._load_yaml(config_path)
        self.guidelines = self._load_yaml(guidelines_path)

        pp_cfg = self.config["preprocessing"]
        self.do_lowercase: bool = pp_cfg.get("lowercase", True)
        self.do_normalize_whitespace: bool = pp_cfg.get(
            "normalize_whitespace", True
        )
        self.strip_stray_numeric_tokens: bool = bool(
            pp_cfg.get("strip_stray_numeric_tokens", True)
        )
        self.promo_noise_patterns: tuple[str, ...] = load_promo_noise_patterns(
            pp_cfg
        )

        # Build ordered abbreviation list — order from config is preserved.
        # Using list of tuples, not dict, to guarantee expansion order.
        # Longer/more specific patterns must come before shorter ones
        # (e.g. "vg++" before "vg+") — enforced in grader.yaml ordering.
        self.abbreviation_pairs: list[tuple[str, str]] = [
            (abbr.lower(), expansion)
            for abbr, expansion in pp_cfg.get("abbreviation_map", {}).items()
        ]

        # Replace the entire abbreviation_patterns block with:
        self.abbreviation_patterns: list[tuple[re.Pattern, str]] = []
        for abbr, expansion in self.abbreviation_pairs:
            escaped = re.escape(abbr.lower())
            if abbr.endswith("+"):
                # Prevent vg+ from matching inside vg++
                # by requiring the next char is not also +
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\+)",
                    re.IGNORECASE,
                )
            else:
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\w)",
                    re.IGNORECASE,
                )
            self.abbreviation_patterns.append((pattern, expansion))

        # Unverified media signals — from config
        self.unverified_signals: list[str] = self.config.get(
            "preprocessing", {}
        ).get(
            "unverified_media_signals",
            [
                "untested",
                "unplayed",
                "sold as seen",
                "haven't played",
                "not played",
                "unable to test",
                "no turntable",
            ],
        )

        # Generic sleeve hard signals — aggregated from every hard-signal
        # variant (legacy ``hard_signals`` plus the strict / cosignal /
        # per-target keys introduced in §13/§13b). Detection here uses
        # substring match, so tier distinctions are irrelevant; callers
        # only care whether *any* Generic hard phrase appears.
        generic_def = self.guidelines.get("grades", {}).get("Generic", {})
        self.generic_signals: list[str] = self._collect_hard_signals(generic_def)

        # Media verifiability cues — used to mark media as unverified when the
        # comment does not include any playback-related language.
        # This is intentionally conservative: we only treat "playback" cues
        # as verifiable, not cosmetic cover wording.
        self._mint_grade_def: dict[str, Any] = (
            self.guidelines.get("grades", {}).get("Mint", {}) or {}
        )
        self.mint_hard_signals: list[str] = self._collect_hard_signals(
            self._mint_grade_def
        )

        media_cue_substrings = (
            "play",
            "played",
            "plays",
            "skip",
            "skipping",
            "surface noise",
            "crackle",
            "crackling",
            "noise",
            "sound",
            "tested",
            "won't play",
            "cannot play",
            "can't play",
        )

        self.media_verifiable_cues: list[str] = []
        # Legacy signal keys (strict/cosignal hard variants are harvested
        # via ``_collect_hard_signals`` below so the §13b migration does
        # not drop Poor's playback cues from the verifiable set).
        _legacy_signal_keys = (
            "supporting_signals",
            "forbidden_signals",
            "supporting_signals_media",
            "forbidden_signals_media",
        )
        for grade_def in self.guidelines.get("grades", {}).values():
            applies_to = grade_def.get("applies_to", [])
            if "media" not in applies_to:
                continue
            candidate_signals: list[str] = list(
                self._collect_hard_signals(grade_def)
            )
            for signal_list_key in _legacy_signal_keys:
                for signal in grade_def.get(signal_list_key, []) or []:
                    if isinstance(signal, str):
                        candidate_signals.append(signal.lower())
            for s in candidate_signals:
                if any(sub in s for sub in media_cue_substrings):
                    self.media_verifiable_cues.append(s)

        # De-duplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for cue in self.media_verifiable_cues:
            if cue in seen:
                continue
            seen.add(cue)
            deduped.append(cue)
        self.media_verifiable_cues = deduped

        # Additional heuristic for comments that explicitly reference the
        # record/media object plus condition defects (not just sleeve).
        self.media_subject_terms: tuple[str, ...] = (
            "vinyl",
            "record",
            "disc",
            "lp",
            "wax",
            "pressing",
            "labels",
            "label",
        )
        self.media_condition_terms: tuple[str, ...] = (
            "mark",
            "marks",
            "scratch",
            "scratches",
            "scuff",
            "scuffs",
            "wear",
            "play wear",
            "surface",
            "dimple",
            "dimples",
            "bubble",
            "bubbling",
            "press",
            "pressed",
        )

        # Protected terms — derived from all hard_signals and
        # supporting_signals across all grades. These must survive
        # all text transformations unchanged.
        self.protected_terms: set[str] = self._build_protected_terms()
        self._protected_term_token_patterns: dict[str, re.Pattern[str]] = {
            t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
            for t in self.protected_terms
            if str(t).strip()
        }

        # Split config
        split_cfg = self.config["data"]["splits"]
        self.train_ratio: float = split_cfg["train"]
        self.val_ratio: float = split_cfg["val"]
        self.test_ratio: float = split_cfg["test"]
        self.random_seed: int = split_cfg.get("random_seed", 42)

        self._harmonization_min_samples: int = int(
            self.config.get("data", {})
            .get("harmonization", {})
            .get("min_samples_per_class", 50)
        )

        # Description adequacy (thin notes — training filter + inference hints)
        da_cfg = pp_cfg.get("description_adequacy") or {}
        self.description_adequacy_enabled: bool = bool(
            da_cfg.get("enabled", False)
        )
        self.drop_insufficient_from_training: bool = bool(
            da_cfg.get("drop_insufficient_from_training", False)
        )
        self.require_both_for_training: bool = bool(
            da_cfg.get("require_both_for_training", True)
        )
        self.min_chars_sleeve_fallback: int = int(
            da_cfg.get("min_chars_sleeve_fallback", 56)
        )
        self.prompt_sleeve: str = str(
            da_cfg.get(
                "user_prompt_sleeve",
                "Add jacket/sleeve condition details.",
            )
        ).strip()
        self.prompt_media: str = str(
            da_cfg.get(
                "user_prompt_media",
                "Describe disc/playable condition or sealed/unplayed.",
            )
        ).strip()
        configured_sleeve_terms = da_cfg.get("sleeve_evidence_terms") or []
        self.sleeve_evidence_terms: tuple[str, ...] = tuple(
            str(t).lower() for t in configured_sleeve_terms
        ) or (
            "jacket",
            "sleeve",
            "cover",
            "gatefold",
            "obi",
            "insert",
            "spine",
            "corner",
            "corners",
            "ringwear",
            "ring wear",
            "seam",
            "split",
            "crease",
            "stain",
            "shrink",
        )
        # Longer phrases first for grade-token detection on cleaned text
        self._grade_phrases: tuple[str, ...] = (
            "very good plus",
            "near mint",
            "mint minus",
            "excellent plus",
            "excellent minus",
            "very good",
            "good plus",
            "excellent",
            "good",
            "mint",
            "poor",
        )

        # Mint sleeve listings often have very short notes ("still sealed", …).
        # When enabled, treat sleeve note as adequate if sleeve_label is Mint and
        # any Mint-ish phrase matches (media label unrestricted).
        self.mint_sleeve_label_relax_sleeve_note: bool = bool(
            da_cfg.get(
                "mint_sleeve_label_relax_sleeve_note",
                da_cfg.get("mint_both_labels_relax_sleeve_note", True),
            )
        )
        _mint_relax: list[str] = list(
            self._collect_hard_signals(self._mint_grade_def)
        )
        _mint_relax_seen: set[str] = set(_mint_relax)
        for _sig in self._mint_grade_def.get("supporting_signals", []) or []:
            if isinstance(_sig, str):
                _ls = _sig.lower().strip()
                if _ls and _ls not in _mint_relax_seen:
                    _mint_relax.append(_ls)
                    _mint_relax_seen.add(_ls)
        for _sig in da_cfg.get("mint_sleeve_note_relax_extra_terms", []) or []:
            if isinstance(_sig, str):
                _ls = _sig.lower().strip()
                if _ls and _ls not in _mint_relax_seen:
                    _mint_relax.append(_ls)
                    _mint_relax_seen.add(_ls)
        for _extra in ("brand new", "like new", "new copy"):
            if _extra not in _mint_relax_seen:
                _mint_relax.append(_extra)
                _mint_relax_seen.add(_extra)
        self.mint_sleeve_relax_substrings: tuple[str, ...] = tuple(_mint_relax)

        # Paths
        processed_dir = Path(self.config["paths"]["processed"])
        splits_dir = Path(self.config["paths"]["splits"])
        self.reports_dir = Path(self.config["paths"]["reports"])
        self.input_path = processed_dir / "unified.jsonl"
        self.output_path = processed_dir / "preprocessed.jsonl"
        self.split_paths = {
            "train": splits_dir / "train.jsonl",
            "val": splits_dir / "val.jsonl",
            "test": splits_dir / "test.jsonl",
            # All inadequate-for-training rows (written when thin-note filter is on)
            "test_thin": splits_dir / "test_thin.jsonl",
        }
        splits_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # MLflow: ``run()`` uses ``mlflow_pipeline_step_run_ctx`` — configure only
        # when a nested step run is actually opened (``log_pipeline_step_runs``).

        # Stats (``process_record`` may run outside ``run()`` — e.g. unit tests).
        self._stats = self._fresh_pipeline_stats()

    @staticmethod
    def _fresh_pipeline_stats() -> dict[str, Any]:
        return {
            "total_processed": 0,
            "protected_terms_lost": 0,
            "generic_text_label_mismatch": 0,
            "sleeve_imbalance_ratio": 0.0,
            "media_imbalance_ratio": 0.0,
            "stratify_key": "",
            "rare_strata_fallback": [],
            "n_adequate_for_training": 0,
            "n_excluded_from_splits": 0,
            "n_test_thin": 0,
        }

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Protected terms
    # -----------------------------------------------------------------------
    @staticmethod
    def _collect_hard_signals(grade_def: dict[str, Any]) -> list[str]:
        """
        Aggregate every hard-signal phrase declared on a grade, across the
        legacy ``hard_signals`` list and the §13/§13b strict/cosignal
        variants (untargeted and per-target ``_sleeve`` / ``_media``
        keys). Deduplicated while preserving first-seen order.

        Used by preprocess-time detectors that care about "is any Generic
        / Mint hard phrase present?" and do not distinguish tiers.
        """
        if not isinstance(grade_def, dict):
            return []
        keys = (
            "hard_signals",
            "hard_signals_strict",
            "hard_signals_cosignal",
            "hard_signals_strict_sleeve",
            "hard_signals_strict_media",
            "hard_signals_cosignal_sleeve",
            "hard_signals_cosignal_media",
        )
        seen: set[str] = set()
        out: list[str] = []
        for key in keys:
            values = grade_def.get(key)
            if not isinstance(values, list):
                continue
            for signal in values:
                if not isinstance(signal, str):
                    continue
                s = signal.lower()
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)
        return out

    def _build_protected_terms(self) -> set[str]:
        """
        Derive protected terms from all grade signal lists in guidelines.
        These terms carry grading signal and must survive normalization.
        Stored in lowercase for case-insensitive comparison.
        """
        grades = self.guidelines.get("grades", {})
        if not isinstance(grades, dict):
            return set()
        return protected_terms_from_grades(grades)

    # -----------------------------------------------------------------------
    # Detection — must run on RAW text
    # -----------------------------------------------------------------------
    def detect_unverified_media(self, text: str) -> bool:
        """
        Returns False (unverifiable) if any unverified media signal
        is found in the raw text. Case-insensitive.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        # 1) Hard unverified signals always win.
        if any(signal in text_lower for signal in self.unverified_signals):
            return False

        # 2) Sealed/Mint exemption: sealed implies Mint media by convention
        # in this project, even if playback isn't described.
        if any(sig in text_lower for sig in self.mint_hard_signals):
            return True

        # 3) If no playback/media cue exists in the comment, mark unverified.
        if any(cue in text_lower for cue in self.media_verifiable_cues):
            return True

        # 4) Comments that mention the media object + defect language
        # (e.g., "Vinyl has light surface marks", "labels bubbling")
        # count as verifiable media descriptions.
        has_media_subject = any(
            term in text_lower for term in self.media_subject_terms
        )
        has_media_condition = any(
            term in text_lower for term in self.media_condition_terms
        )
        if has_media_subject and has_media_condition:
            return True

        # 5) Explicit play-testing language (may not appear in guideline-derived cues).
        play_markers = (
            "plays perfectly",
            "plays great",
            "plays fine",
            "plays well",
            "plays cleanly",
            "plays through",
            "tested",
            "spin tested",
        )
        if any(m in text_lower for m in play_markers):
            return True

        return False

    def detect_media_evidence_strength(self, text: str) -> str:
        """
        Estimate how directly the comment describes playable media condition.

        Returns one of:
          - "none": no usable media evidence
          - "weak": indirect/limited media evidence
          - "strong": clear playback/media-condition evidence
        """
        text_lower = text.lower()
        if any(signal in text_lower for signal in self.unverified_signals):
            return "none"

        playback_hits = sum(
            1 for cue in self.media_verifiable_cues if cue in text_lower
        )
        has_media_subject = any(
            term in text_lower for term in self.media_subject_terms
        )
        has_media_condition = any(
            term in text_lower for term in self.media_condition_terms
        )

        if playback_hits >= 2:
            return "strong"
        if playback_hits >= 1 and (has_media_subject or has_media_condition):
            return "strong"
        if has_media_subject and has_media_condition:
            return "weak"
        if any(sig in text_lower for sig in self.mint_hard_signals):
            # Sealed/unopened gives a convention-based high media label,
            # but textual evidence for actual playback condition is limited.
            return "weak"
        return "none"

    def detect_generic_sleeve(self, text: str) -> bool:
        """
        Returns True if any Generic hard signal is found in raw text.
        Used to re-confirm Generic sleeve labels from text alone.

        Note: sleeve_label may already be Generic from ingestion.
        This is a secondary text-based detection for records where
        the API condition field was ambiguous.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        return any(signal in text_lower for signal in self.generic_signals)

    def _count_distinct_grade_phrases(self, text_clean_lower: str) -> int:
        """How many canonical grade phrases appear (after abbreviation expansion)."""
        return sum(1 for phrase in self._grade_phrases if phrase in text_clean_lower)

    def sleeve_note_adequate(self, text_clean: str) -> bool:
        """
        True if the note plausibly describes jacket/sleeve/packaging,
        or uses multi-grade shorthand (e.g. NM/VG), or is long free text.
        """
        t = text_clean.strip().lower()
        if not t:
            return False
        if any(hint in t for hint in self.sleeve_evidence_terms):
            return True
        if self._count_distinct_grade_phrases(t) >= 2:
            return True
        if len(text_clean.strip()) >= self.min_chars_sleeve_fallback:
            return True
        return False

    def media_note_adequate(self, raw_text: str) -> bool:
        """True when media_evidence_strength is not ``none`` (see detect_*)."""
        return self.detect_media_evidence_strength(raw_text) != "none"

    def _mint_listing_sleeve_relaxed_ok(
        self, raw_text: str, text_clean: str
    ) -> bool:
        """Short sealed / Mint-ish copy counts as sleeve evidence."""
        blob = f"{raw_text} {text_clean}".lower()
        return any(h in blob for h in self.mint_sleeve_relax_substrings)

    def compute_description_quality(
        self,
        raw_text: str,
        text_clean: str,
        *,
        sleeve_label: str | None = None,
        media_label: str | None = None,
    ) -> dict:
        """
        Fields for training filter and inference UX.

        Returns keys: sleeve_note_adequate, media_note_adequate,
        adequate_for_training, description_quality_gaps (list[str]),
        description_quality_prompts (list[str]), needs_richer_note (bool).

        When ``mint_sleeve_label_relax_sleeve_note`` is enabled in config and
        ``sleeve_label`` is ``Mint``, sleeve adequacy also passes if the note
        contains Mint listing phrases (sealed, shrink, brand new, …).
        """
        if not self.description_adequacy_enabled:
            return {
                "sleeve_note_adequate": True,
                "media_note_adequate": True,
                "adequate_for_training": True,
                "description_quality_gaps": [],
                "description_quality_prompts": [],
                "needs_richer_note": False,
            }

        sleeve_ok = self.sleeve_note_adequate(text_clean)
        if (
            not sleeve_ok
            and self.mint_sleeve_label_relax_sleeve_note
            and str(sleeve_label or "").strip() == "Mint"
            and self._mint_listing_sleeve_relaxed_ok(raw_text, text_clean)
        ):
            sleeve_ok = True
        media_ok = self.media_note_adequate(raw_text)
        gaps: list[str] = []
        prompts: list[str] = []
        if not sleeve_ok:
            gaps.append("sleeve")
            prompts.append(self.prompt_sleeve)
        if not media_ok:
            gaps.append("media")
            prompts.append(self.prompt_media)

        if self.require_both_for_training:
            train_ok = sleeve_ok and media_ok
        else:
            train_ok = sleeve_ok or media_ok

        return {
            "sleeve_note_adequate": sleeve_ok,
            "media_note_adequate": media_ok,
            "adequate_for_training": train_ok,
            "description_quality_gaps": gaps,
            "description_quality_prompts": prompts,
            "needs_richer_note": bool(gaps),
        }

    # -----------------------------------------------------------------------
    # Text normalization
    # -----------------------------------------------------------------------
    def _lowercase(self, text: str) -> str:
        return text.lower() if self.do_lowercase else text

    def _normalize_whitespace(self, text: str) -> str:
        if not self.do_normalize_whitespace:
            return text
        # Collapse multiple whitespace characters into a single space
        # and strip leading/trailing whitespace
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _strip_leading_numeric_boilerplate(text: str) -> str:
        """
        Drop a lone leading catalog/index digit before obvious condition
        boilerplate (e.g. ``6 sealed, new hype sticker`` → ``sealed, …``).
        Does not strip counts that are part of a format description
        (``2 lp``, ``7\"``, ``disk 2 of 3``, ``6 inch``, ``3 inches``).
        """
        s = text.strip()
        m = re.match(
            r"^(\d{1,2})\s+",
            s,
            flags=re.IGNORECASE,
        )
        if not m:
            return text
        rest = s[m.end() :].lstrip()
        rest_lower = rest.lower()
        if re.match(r'^(?:\d{1,2}\s*["\u201d]|\d+\s*/\d+)', rest_lower):
            return text
        if re.match(
            r"^(?:\d+\s+lp\b|\d+\s+inch\b|\d+\s+inches\b|disk\s+\d+\s+of\b)",
            rest_lower,
        ):
            return text
        if re.match(
            r"^(?:sealed|new\s+hype|new\b|nm\b|mint\b|vg\+|vg\b|ex\b|poor\b|good\b)",
            rest_lower,
        ):
            return rest
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations using ordered regex patterns.
        Order is preserved from grader.yaml abbreviation_map.
        Longer patterns (vg++) are applied before shorter ones (vg+)
        to prevent partial match corruption.

        When an abbreviation is immediately followed by a letter (e.g.
        ``vg+original`` with no space), append a single space so the
        expansion does not glue to the next word (``… plus original``).
        """
        for pattern, expansion in self.abbreviation_patterns:

            def _repl(m: re.Match[str], exp: str = expansion) -> str:
                s = m.string
                j = m.end()
                if j < len(s) and s[j].isalpha():
                    return f"{exp} "
                return exp

            text = pattern.sub(_repl, text)
        return text

    def _verify_protected_terms(
        self, original: str, cleaned: str
    ) -> list[str]:
        """
        Sanity check — verify that protected terms present in the
        original text as **whole tokens** still appear that way in the cleaned
        text (``\\b`` word boundaries + ``re.escape`` per term).

        Returns list of terms that were lost during transformation.
        A non-empty list indicates a preprocessing bug or an aggressive strip
        that removed a real defect token.
        """
        lost: list[str] = []
        for term, pat in self._protected_term_token_patterns.items():
            if pat.search(original) and not pat.search(cleaned):
                lost.append(term)
        return lost

    def clean_text(self, text: str) -> str:
        """
        Apply full text normalization pipeline in correct order.
        Runs on text AFTER detection steps have completed.

        Steps:
          1. Lowercase
          2. Normalize whitespace
          3. Strip listing promo / shipping boilerplate
          4. Optionally strip leading catalog digit (``strip_stray_numeric_tokens``)
          5. Expand abbreviations
        """
        cleaned = self._lowercase(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = strip_listing_promo_noise(
            cleaned,
            self.promo_noise_patterns,
            protected_term_patterns=self._protected_term_token_patterns,
        )
        if self.strip_stray_numeric_tokens:
            cleaned = self._strip_leading_numeric_boilerplate(cleaned)
        cleaned = self._expand_abbreviations(cleaned)
        return cleaned

    @classmethod
    def normalize_text_for_tfidf(
        cls,
        text: str,
        *,
        preprocessing_cfg: dict[str, Any],
        protected_term_patterns: dict[str, re.Pattern[str]] | None = None,
    ) -> str:
        """
        Match ``clean_text`` through promo stripping and leading-digit cleanup,
        but **omit** abbreviation expansion so TF-IDF sees the same tokens as
        ``text_clean`` from preprocess (which already expanded abbrevs).

        Pass the same ``protected_term_patterns`` as ``Preprocessor`` uses
        (from ``build_protected_term_token_patterns(guidelines)``) so TF-IDF
        skips the same gated ``###`` / ``[]`` / ``***`` spans as training.
        """
        pp = preprocessing_cfg
        s = text.strip()
        if pp.get("lowercase", True):
            s = s.lower()
        if pp.get("normalize_whitespace", True):
            s = re.sub(r"\s+", " ", s).strip()
        phrases = load_promo_noise_patterns(pp)
        s = strip_listing_promo_noise(
            s, phrases, protected_term_patterns=protected_term_patterns
        )
        if bool(pp.get("strip_stray_numeric_tokens", True)):
            s = cls._strip_leading_numeric_boilerplate(s)
        return s

    # -----------------------------------------------------------------------
    # Record processing
    # -----------------------------------------------------------------------
    def process_record(self, record: dict) -> dict:
        """
        Process a single unified record. Returns a new dict with:
          - text_clean:       normalized, expanded text
          - media_verifiable: re-detected from raw text
          - All original fields preserved unchanged

        Detection runs on original text.
        Cleaning runs after detection.
        """
        raw_text = record.get("text", "")

        # Step 1 & 2: detection on raw text
        media_verifiable = self.detect_unverified_media(raw_text)
        media_evidence_strength = self.detect_media_evidence_strength(raw_text)
        text_based_generic = self.detect_generic_sleeve(raw_text)

        # Step 3-5: text normalization
        text_clean = self.clean_text(raw_text)

        # Step 6: protected term sanity check
        lost_terms = self._verify_protected_terms(raw_text, text_clean)
        if lost_terms:
            logger.warning(
                "Protected terms lost during cleaning for item_id=%s: %s",
                record.get("item_id", "?"),
                lost_terms,
            )
            self._stats["protected_terms_lost"] += 1

        # Build output record — original fields preserved, new fields appended
        processed = {**record}
        processed["text_clean"] = text_clean
        processed["media_verifiable"] = media_verifiable
        processed["media_evidence_strength"] = media_evidence_strength

        dq = self.compute_description_quality(
            raw_text,
            text_clean,
            sleeve_label=str(record.get("sleeve_label") or ""),
            media_label=str(record.get("media_label") or ""),
        )
        processed.update(dq)

        # If text-based Generic detection fires but sleeve_label is not
        # already Generic, log for review — do not silently override label.
        # Label integrity is paramount; discrepancies are flagged, not fixed.
        if text_based_generic and record.get("sleeve_label") != "Generic":
            logger.debug(
                "Generic signal in text but sleeve_label=%r for item_id=%s. "
                "Label preserved — review may be needed.",
                record.get("sleeve_label"),
                record.get("item_id", "?"),
            )
            self._stats["generic_text_label_mismatch"] += 1

        return processed

    # -----------------------------------------------------------------------
    # Adaptive stratification
    # -----------------------------------------------------------------------
    def _compute_imbalance(self, labels: list[str]) -> float:
        """
        Compute imbalance ratio for a label list.
        Ratio = max_class_count / min_class_count.
        Higher ratio = more imbalanced.
        """
        counts = Counter(labels)
        if len(counts) < 2:
            return 1.0
        return max(counts.values()) / min(counts.values())

    def select_stratify_key(self, records: list[dict]) -> str:
        """
        Adaptively select the stratification key based on which
        target has higher class imbalance.

        Logs the decision and both imbalance ratios to MLflow.
        """
        sleeve_labels = [r["sleeve_label"] for r in records]
        media_labels = [r["media_label"] for r in records]

        sleeve_imbalance = self._compute_imbalance(sleeve_labels)
        media_imbalance = self._compute_imbalance(media_labels)

        stratify_key = (
            "sleeve_label"
            if sleeve_imbalance >= media_imbalance
            else "media_label"
        )

        logger.info(
            "Adaptive stratification — sleeve imbalance: %.2f | "
            "media imbalance: %.2f | stratifying on: %s",
            sleeve_imbalance,
            media_imbalance,
            stratify_key,
        )

        self._stats["sleeve_imbalance_ratio"] = sleeve_imbalance
        self._stats["media_imbalance_ratio"] = media_imbalance
        self._stats["stratify_key"] = stratify_key

        return stratify_key

    # -----------------------------------------------------------------------
    # Train/val/test split
    # -----------------------------------------------------------------------
    def split_records(self, records: list[dict]) -> dict[str, list[dict]]:
        """
        Assign train/val/test split to each record using adaptive
        stratified sampling.

        Strategy:
          1. Select stratification key based on imbalance ratio
          2. Attempt stratified split using StratifiedShuffleSplit
          3. If any stratum has < 2 samples, fall back to random split
             for affected records and log a warning

        Returns dict mapping split name → list of records.
        Each record has a "split" field added.
        """
        stratify_key = self.select_stratify_key(records)
        labels = [r[stratify_key] for r in records]

        # Identify strata with fewer than 2 samples — cannot be stratified
        label_counts = Counter(labels)
        too_rare = {
            label for label, count in label_counts.items() if count < 2
        }

        if too_rare:
            logger.warning(
                "Strata with < 2 samples — falling back to random split "
                "for these classes: %s",
                too_rare,
            )
            self._stats["rare_strata_fallback"] = list(too_rare)

        # Separate rare and stratifiable records
        rare_records = [r for r in records if r[stratify_key] in too_rare]
        strat_records = [r for r in records if r[stratify_key] not in too_rare]
        strat_labels = [r[stratify_key] for r in strat_records]

        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

        if strat_records:
            # First split: train vs (val + test)
            val_test_ratio = self.val_ratio + self.test_ratio
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_test_ratio,
                random_state=self.random_seed,
            )
            train_idx, val_test_idx = next(
                splitter.split(strat_records, strat_labels)
            )

            train_records = [strat_records[i] for i in train_idx]
            val_test_records = [strat_records[i] for i in val_test_idx]
            val_test_labels = [strat_labels[i] for i in val_test_idx]

            # Second split: val vs test from the val_test pool
            val_ratio_adjusted = self.val_ratio / val_test_ratio
            splitter2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - val_ratio_adjusted,
                random_state=self.random_seed,
            )
            val_idx, test_idx = next(
                splitter2.split(val_test_records, val_test_labels)
            )

            splits["train"] = train_records
            splits["val"] = [val_test_records[i] for i in val_idx]
            splits["test"] = [val_test_records[i] for i in test_idx]

        # Distribute rare records proportionally using random assignment
        if rare_records:
            import random

            rng = random.Random(self.random_seed)
            for record in rare_records:
                split_name = rng.choices(
                    ["train", "val", "test"],
                    weights=[
                        self.train_ratio,
                        self.val_ratio,
                        self.test_ratio,
                    ],
                )[0]
                splits[split_name].append(record)

        # Tag each record with its split name
        for split_name, split_records in splits.items():
            for record in split_records:
                record["split"] = split_name

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

        return splits

    # -----------------------------------------------------------------------
    # Loading and saving
    # -----------------------------------------------------------------------
    def load_unified(self) -> list[dict]:
        """Load the unified JSONL file produced by harmonize_labels.py."""
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Unified dataset not found at {self.input_path}. "
                "Run harmonize_labels.py first."
            )
        records = []
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d records from %s", len(records), self.input_path)
        return records

    def save_preprocessed(self, records: list[dict]) -> None:
        """Write full preprocessed dataset with split field to JSONL."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Saved %d preprocessed records to %s",
            len(records),
            self.output_path,
        )

    def save_splits(self, splits: dict[str, list[dict]]) -> None:
        """Write individual train/val/test JSONL files."""
        for split_name, split_records in splits.items():
            path = self.split_paths[split_name]
            with open(path, "w", encoding="utf-8") as f:
                for record in split_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Saved %d records to %s", len(split_records), path)

    def _write_test_thin_jsonl(self, thin_records: list[dict]) -> None:
        """Eval-only split: rows not eligible for train/val/test adequacy pool."""
        path = self.split_paths["test_thin"]
        with open(path, "w", encoding="utf-8") as f:
            for record in thin_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Saved %d thin-note eval records to %s",
            len(thin_records),
            path,
        )

    def _remove_stale_test_thin_file(self) -> None:
        path = self.split_paths["test_thin"]
        if path.exists():
            path.unlink()
            logger.info(
                "Removed %s — thin-note split only when "
                "description_adequacy + drop_insufficient_from_training are on",
                path,
            )

    # -----------------------------------------------------------------------
    # Class distribution report (post-preprocess, by split)
    # -----------------------------------------------------------------------
    @staticmethod
    def _label_distribution(
        records: list[dict],
    ) -> dict[str, dict[str, int]]:
        sleeve: Counter[str] = Counter()
        media: Counter[str] = Counter()
        for record in records:
            sleeve[str(record["sleeve_label"])] += 1
            media[str(record["media_label"])] += 1
        return {"sleeve": dict(sleeve), "media": dict(media)}

    def _rare_class_warnings_for_dist(
        self,
        distribution: dict[str, dict[str, int]],
        *,
        scope: str,
    ) -> list[str]:
        warnings: list[str] = []
        threshold = self._harmonization_min_samples
        for target, grade_counts in distribution.items():
            for grade, count in grade_counts.items():
                if count < threshold:
                    warnings.append(
                        f"RARE CLASS — scope: {scope}, target: {target}, "
                        f"grade: {grade}, count: {count} "
                        f"(threshold: {threshold})"
                    )
        return warnings

    def _format_grade_table_lines(
        self,
        distribution: dict[str, dict[str, int]],
    ) -> list[str]:
        sleeve_order = self.guidelines["sleeve_grades"]
        sleeve_dist = distribution["sleeve"]
        media_dist = distribution["media"]
        lines = [
            "-" * 60,
            f"{'Grade':<20} {'Sleeve':>8} {'Media':>8}",
            "-" * 60,
        ]
        for grade in sleeve_order:
            sleeve_count = sleeve_dist.get(grade, 0)
            media_count = (
                "-" if grade == "Generic" else media_dist.get(grade, 0)
            )
            lines.append(
                f"{grade:<20} {sleeve_count:>8} {str(media_count):>8}"
            )
        sleeve_total = sum(sleeve_dist.values())
        media_total = sum(media_dist.values())
        lines += [
            "-" * 60,
            f"{'Total':<20} {sleeve_total:>8} {media_total:>8}",
            "",
        ]
        return lines

    def _format_class_distribution_splits_report(
        self,
        processed: list[dict],
        out_splits: dict[str, list[dict]],
    ) -> str:
        lines: list[str] = [
            "=" * 60,
            "VINYL GRADER — CLASS DISTRIBUTION BY SPLIT (AFTER PREPROCESS)",
            "=" * 60,
            "",
            f"Total preprocessed rows (full pool): {len(processed):>10}",
        ]
        if (
            self.description_adequacy_enabled
            and self.drop_insufficient_from_training
        ):
            eligible = self._stats.get("n_adequate_for_training", 0)
            excl = self._stats.get("n_excluded_from_splits", 0)
            lines += [
                f"Eligible for train/val/test:       {eligible:>10}",
                f"Excluded from splits (thin):     {excl:>10}",
                "",
            ]

        by_source: Counter[str] = Counter(
            str(r.get("source") or "?") for r in processed
        )
        lines.append("By source (full pool):")
        for src in sorted(by_source.keys()):
            lines.append(f"  {src + ':':<40} {by_source[src]:>10}")
        lines.append("")
        lines.append("Split sizes:")
        for name in ("train", "val", "test", "test_thin"):
            if name not in out_splits:
                continue
            lines.append(f"  {name + ':':<40} {len(out_splits[name]):>10}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("Full pool (all rows written to preprocessed.jsonl)")
        lines.append("-" * 60)
        full_dist = self._label_distribution(processed)
        lines.extend(self._format_grade_table_lines(full_dist))
        all_warnings = self._rare_class_warnings_for_dist(
            full_dist, scope="full_pool"
        )

        for split_name in ("train", "val", "test", "test_thin"):
            if split_name not in out_splits:
                continue
            rows = out_splits[split_name]
            lines.append("-" * 60)
            lines.append(f"Split: {split_name} ({len(rows)} rows)")
            lines.append("-" * 60)
            dist = self._label_distribution(rows)
            lines.extend(self._format_grade_table_lines(dist))
            all_warnings.extend(
                self._rare_class_warnings_for_dist(
                    dist, scope=f"split:{split_name}"
                )
            )

        if all_warnings:
            lines += [
                "=" * 60,
                "RARE CLASS WARNINGS",
                "=" * 60,
            ]
            for w in all_warnings:
                lines.append(f"  {w}")
            lines.append("")

        lines += [
            "=" * 60,
            "Note: Poor and Generic are expected to be rare.",
            "Rule engine owns these grades — low sample count",
            "does not prevent grading of these conditions.",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _save_class_distribution_splits_report(
        self,
        processed: list[dict],
        out_splits: dict[str, list[dict]],
    ) -> None:
        path = self.reports_dir / "class_distribution_splits.txt"
        text = self._format_class_distribution_splits_report(
            processed, out_splits
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved class distribution (splits) to %s", path)

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(self, splits: dict[str, list[dict]]) -> None:
        mlflow.log_params(
            {
                "lowercase": self.do_lowercase,
                "normalize_whitespace": self.do_normalize_whitespace,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "random_seed": self.random_seed,
                "stratify_key": self._stats.get("stratify_key", "unknown"),
                "n_abbreviations": len(self.abbreviation_pairs),
            }
        )
        mlflow.log_metrics(
            {
                "total_processed": self._stats["total_processed"],
                "protected_terms_lost": self._stats["protected_terms_lost"],
                "generic_text_label_mismatch": self._stats[
                    "generic_text_label_mismatch"
                ],
                "sleeve_imbalance_ratio": self._stats[
                    "sleeve_imbalance_ratio"
                ],
                "media_imbalance_ratio": self._stats["media_imbalance_ratio"],
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(splits["test"]),
                "n_adequate_for_training": self._stats.get(
                    "n_adequate_for_training", 0
                ),
                "n_excluded_from_splits": self._stats.get(
                    "n_excluded_from_splits", 0
                ),
                "n_test_thin": self._stats.get("n_test_thin", 0),
            }
        )

    def _save_description_adequacy_report(
        self,
        all_processed: list[dict],
        split_pool: list[dict],
    ) -> None:
        path = self.reports_dir / "description_adequacy_summary.txt"
        excl = [r for r in all_processed if not r["adequate_for_training"]]
        lines = [
            "Description adequacy (preprocessing)",
            "=" * 60,
            f"Total records:           {len(all_processed)}",
            f"Eligible for splits:     {len(split_pool)}",
            f"Excluded (thin notes):   {len(excl)}",
            "",
            "Excluded rows lack sleeve cues and/or playable-media cues "
            "(see preprocessing.description_adequacy in grader.yaml).",
            "They remain in preprocessed.jsonl with adequacy flags for audit.",
            "Eval-only split: grader/data/splits/test_thin.jsonl (same rows).",
            "",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Wrote %s", path)

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> dict[str, list[dict]]:
        """
        Full preprocessing pipeline:
          1. Load unified.jsonl
          2. Process each record (detect + clean)
          3. Adaptive stratified split (adequate rows only when thin filter on)
          4. Save preprocessed.jsonl, train/val/test, and optional test_thin.jsonl
          5. Save ``class_distribution_splits.txt`` under ``paths.reports``
          6. Log metrics to MLflow

        Args:
            dry_run: process and split but do not write files
                     or log to MLflow.

        Returns:
            Dict mapping split name → list of processed records.
        """
        self._stats = self._fresh_pipeline_stats()

        with mlflow_pipeline_step_run_ctx(self.config, "preprocess") as mlf:
            records = self.load_unified()

            # Process each record
            processed: list[dict] = []
            for record in records:
                processed.append(self.process_record(record))
                self._stats["total_processed"] += 1

            logger.info(
                "Processed %d records. Protected term losses: %d. "
                "Generic/label mismatches: %d.",
                self._stats["total_processed"],
                self._stats["protected_terms_lost"],
                self._stats["generic_text_label_mismatch"],
            )

            split_pool = processed
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                split_pool = [r for r in processed if r["adequate_for_training"]]
                self._stats["n_excluded_from_splits"] = len(processed) - len(
                    split_pool
                )
                self._stats["n_adequate_for_training"] = len(split_pool)
                logger.info(
                    "Description adequacy — eligible for splits: %d | "
                    "excluded (thin notes): %d",
                    len(split_pool),
                    self._stats["n_excluded_from_splits"],
                )
                self._save_description_adequacy_report(processed, split_pool)
            else:
                self._stats["n_adequate_for_training"] = len(processed)

            if not split_pool:
                raise ValueError(
                    "No records left for train/val/test splits after "
                    "description_adequacy filtering. Relax "
                    "preprocessing.description_adequacy or set "
                    "drop_insufficient_from_training: false."
                )

            # Adaptive stratified split (training-eligible rows only when filtering)
            splits = self.split_records(split_pool)

            thin_records: list[dict] = []
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                thin_records = [
                    r for r in processed if not r["adequate_for_training"]
                ]
                for r in thin_records:
                    r["split"] = "test_thin"
                self._stats["n_test_thin"] = len(thin_records)
            else:
                self._remove_stale_test_thin_file()

            out_splits = dict(splits)
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                out_splits["test_thin"] = thin_records

            if dry_run:
                logger.info(
                    "Dry run — skipping file writes and MLflow logging."
                )
                return out_splits

            # Save outputs
            self.save_preprocessed(processed)
            self.save_splits(splits)
            self._write_test_thin_jsonl(thin_records)
            self._save_class_distribution_splits_report(processed, out_splits)
            if mlf:
                self._log_mlflow(splits)

        return out_splits
