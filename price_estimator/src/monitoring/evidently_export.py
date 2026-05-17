"""Optional Evidently HTML report for human review (non-authoritative vs drift_stats)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_data_drift_html(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write Evidently ``DataDriftPreset`` HTML (requires ``evidently`` installed)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snap = Report([DataDriftPreset()]).run(
        current_data=current,
        reference_data=reference,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(out_path))
