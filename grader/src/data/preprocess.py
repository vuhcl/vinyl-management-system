"""
grader/src/data/preprocess.py

Text preprocessing pipeline for the vinyl condition grader.
Reads unified.jsonl, applies text normalization and abbreviation
expansion, detects unverified media signals, assigns train/val/test
splits using adaptive stratification, and writes output JSONL files.

Transformation order (strictly enforced):
  1. Detect unverified media signals  — on raw text
  2. Detect Generic sleeve signals    — on raw text
  3. Lowercase
  4. Normalize whitespace
  5. Expand abbreviations             — after lowercase
  6. Verify protected terms survive   — sanity check

The original `text` field is preserved. Cleaned text is written
to a new `text_clean` field. Labels are never modified.

Usage:
    python -m grader.src.data.preprocess
    python -m grader.src.data.preprocess --dry-run
"""

import copy
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import mlflow
import yaml

from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_pipeline_step_run_ctx,
)
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
        grades.Generic.hard_signals       — for Generic sleeve detection
        grades[*].hard_signals
            — protected terms derived from all signals
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

        # Generic sleeve hard signals — from guidelines
        self.generic_signals: list[str] = [
            s.lower()
            for s in self.guidelines["grades"]["Generic"]["hard_signals"]
        ]

        # Media verifiability cues — used to mark media as unverified when the
        # comment does not include any playback-related language.
        # This is intentionally conservative: we only treat "playback" cues
        # as verifiable, not cosmetic cover wording.
        mint_def = self.guidelines.get("grades", {}).get("Mint", {})
        self.mint_hard_signals: list[str] = [
            s.lower()
            for s in mint_def.get("hard_signals", [])
            if isinstance(s, str)
        ]

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
        for grade_def in self.guidelines.get("grades", {}).values():
            applies_to = grade_def.get("applies_to", [])
            if "media" not in applies_to:
                continue
            for signal_list_key in [
                "hard_signals",
                "supporting_signals",
                "forbidden_signals",
            ]:
                for signal in grade_def.get(signal_list_key, []):
                    if not isinstance(signal, str):
                        continue
                    s = signal.lower()
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

        # Split config
        split_cfg = self.config["data"]["splits"]
        self.train_ratio: float = split_cfg["train"]
        self.val_ratio: float = split_cfg["val"]
        self.test_ratio: float = split_cfg["test"]
        self.random_seed: int = split_cfg.get("random_seed", 42)

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

        if mlflow_enabled(self.config):
            configure_mlflow_from_config(self.config)

        # Stats
        self._stats: dict = {}

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
    def _build_protected_terms(self) -> set[str]:
        """
        Derive protected terms from all grade signal lists in guidelines.
        These terms carry grading signal and must survive normalization.
        Stored in lowercase for case-insensitive comparison.
        """
        terms: set[str] = set()
        grades = self.guidelines.get("grades", {})
        for grade_def in grades.values():
            for signal_list_key in [
                "hard_signals",
                "supporting_signals",
                "forbidden_signals",
            ]:
                for signal in grade_def.get(signal_list_key, []):
                    terms.add(signal.lower())
        return terms

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

    def compute_description_quality(
        self, raw_text: str, text_clean: str
    ) -> dict:
        """
        Fields for training filter and inference UX.

        Returns keys: sleeve_note_adequate, media_note_adequate,
        adequate_for_training, description_quality_gaps (list[str]),
        description_quality_prompts (list[str]), needs_richer_note (bool).
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

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations using ordered regex patterns.
        Order is preserved from grader.yaml abbreviation_map.
        Longer patterns (vg++) are applied before shorter ones (vg+)
        to prevent partial match corruption.
        """
        for pattern, expansion in self.abbreviation_patterns:
            text = pattern.sub(expansion, text)
        return text

    def _verify_protected_terms(
        self, original: str, cleaned: str
    ) -> list[str]:
        """
        Sanity check — verify that protected terms present in the
        original text are still present in the cleaned text.

        Returns list of terms that were lost during transformation.
        A non-empty list indicates a preprocessing bug.
        """
        original_lower = original.lower()
        lost = []
        for term in self.protected_terms:
            if term in original_lower and term not in cleaned.lower():
                lost.append(term)
        return lost

    def clean_text(self, text: str) -> str:
        """
        Apply full text normalization pipeline in correct order.
        Runs on text AFTER detection steps have completed.

        Steps:
          1. Lowercase
          2. Normalize whitespace
          3. Expand abbreviations
        """
        cleaned = self._lowercase(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._expand_abbreviations(cleaned)
        return cleaned

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

        dq = self.compute_description_quality(raw_text, text_clean)
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
          5. Log metrics to MLflow

        Args:
            dry_run: process and split but do not write files
                     or log to MLflow.

        Returns:
            Dict mapping split name → list of processed records.
        """
        self._stats = {
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
            if mlf:
                self._log_mlflow(splits)

        return out_splits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess unified vinyl grader dataset"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and split without writing output files",
    )
    args = parser.parse_args()

    preprocessor = Preprocessor(
        config_path=args.config,
        guidelines_path=args.guidelines,
    )
    preprocessor.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
