"""Composed :class:`Preprocessor` for the vinyl condition grader."""

from __future__ import annotations

from .cleaning_mixin import PreprocessorCleaningMixin
from .detection_mixin import PreprocessorDetectionMixin
from .init_mixin import PreprocessorInitMixin
from .io_mixin import PreprocessorIOMixin
from .record_mixin import PreprocessorRecordMixin
from .reports_mixin import PreprocessorReportsMixin
from .run_mixin import PreprocessorRunMixin
from .signals_mixin import PreprocessorSignalsMixin
from .split_mixin import PreprocessorSplitMixin


class Preprocessor(
    PreprocessorInitMixin,
    PreprocessorSignalsMixin,
    PreprocessorDetectionMixin,
    PreprocessorCleaningMixin,
    PreprocessorRecordMixin,
    PreprocessorSplitMixin,
    PreprocessorIOMixin,
    PreprocessorReportsMixin,
    PreprocessorRunMixin,
):
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

