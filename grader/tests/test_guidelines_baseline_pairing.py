"""
Integration gate: committed ``rule_engine_baseline.json`` must match
``grading_guidelines.yaml`` when both define ``guidelines_version``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUIDELINES = _REPO_ROOT / "grader" / "configs" / "grading_guidelines.yaml"
_BASELINE = _REPO_ROOT / "grader" / "reports" / "rule_engine_baseline.json"


@pytest.mark.skipif(not _BASELINE.is_file(), reason="no committed baseline JSON")
def test_committed_baseline_matches_guidelines_version() -> None:
    with open(_GUIDELINES, encoding="utf-8") as f:
        g = yaml.safe_load(f)
    assert isinstance(g, dict)
    y_ver = g.get("guidelines_version")
    assert y_ver is not None and str(y_ver).strip(), (
        "grading_guidelines.yaml must define guidelines_version"
    )
    with open(_BASELINE, encoding="utf-8") as f:
        b = json.load(f)
    b_ver = b.get("guidelines_version")
    assert b_ver is not None, (
        "rule_engine_baseline.json should include guidelines_version "
        "(regenerate via pipeline eval after rubric changes)"
    )
    assert str(b_ver).strip() == str(y_ver).strip(), (
        f"baseline guidelines_version {b_ver!r} != YAML {y_ver!r}"
    )
