"""Preprocessor configuration and constructor."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Optional

from grader.src.config_io import load_yaml_mapping

from ..listing_promo import load_promo_noise_patterns


class PreprocessorInitMixin:
    def __init__(
        self,
        config_path: str,
        guidelines_path: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = load_yaml_mapping(config_path)
        self.guidelines = load_yaml_mapping(guidelines_path)

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
        self.generic_signals: list[str] = self._collect_hard_signals(
            generic_def
        )

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
        # When enabled, treat sleeve note as adequate if sleeve_label is Mint
        # and any Mint-ish phrase matches (media label unrestricted).
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
            # Inadequate-for-training rows (written when thin-note filter is on)
            "test_thin": splits_dir / "test_thin.jsonl",
        }
        splits_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # MLflow: ``run()`` uses ``mlflow_pipeline_step_run_ctx`` — configure
        # only when a nested step run is actually opened
        # (``log_pipeline_step_runs``).

        # Stats (``process_record`` may run outside ``run()`` — e.g. unit tests).
        self._stats = self._fresh_pipeline_stats()
