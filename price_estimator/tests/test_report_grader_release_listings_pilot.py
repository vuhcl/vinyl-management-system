"""Smoke test for report_grader_release_listings_pilot script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_pilot_script_runs(tmp_path) -> None:
    repo = Path(__file__).resolve().parents[2]
    jl = tmp_path / "rows.jsonl"
    rec = {
        "item_id": "1",
        "source": "discogs",
        "text": (
            "Ring wear on cover corners; vinyl plays cleanly with faint surface noise."
        ),
        "sleeve_label": "Very Good",
        "media_label": "Near Mint",
        "label_confidence": 1.0,
    }
    jl.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    script = repo / "price_estimator" / "scripts" / "report_grader_release_listings_pilot.py"
    cfg = repo / "grader" / "configs" / "grader.yaml"
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(repo)
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--jsonl",
            str(jl),
            "--config",
            str(cfg),
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "rows=1" in r.stdout
    assert "adequate_for_training=" in r.stdout
