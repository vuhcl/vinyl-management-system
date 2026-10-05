"""Canonical feature-pipeline fixtures (column order + vector checksum).

Regenerate checksums after intentional featurizer changes:

  VINYLIQ_REGEN_FEATURE_FIXTURES=1 uv run pytest \\
    price_estimator/tests/test_feature_pipeline_fixtures.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from price_estimator.src.features.vinyliq_features import (
    residual_training_feature_columns,
    row_dict_for_inference,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "feature_pipeline"
CASE_FILES = (
    "nm_synthetic.json",
    "vg_synthetic.json",
    "cold_start_synthetic.json",
)


def _vector_sha16(row: dict[str, float], cols: list[str]) -> str:
    vec = [round(float(row[c]), 6) for c in cols]
    payload = json.dumps(vec, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _build_row(case: dict[str, Any]) -> dict[str, float]:
    idx = case["indices"]
    residual = bool(case.get("residual_mode", True))
    return row_dict_for_inference(
        str(case["release_id"]),
        case.get("media_condition"),
        case.get("sleeve_condition"),
        dict(case["stats"]),
        dict(case["catalog"]),
        genre_index=float(idx["genre_index"]),
        country_index=float(idx["country_index"]),
        primary_artist_index=float(idx["primary_artist_index"]),
        primary_label_index=float(idx["primary_label_index"]),
        include_marketplace_scalars_in_features=not residual,
    )


def _load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for name in CASE_FILES:
        path = FIXTURE_DIR / name
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def _regen_fixture_file(path: Path, case: dict[str, Any], digest: str) -> None:
    updated = dict(case)
    updated["vector_sha16"] = digest
    path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("case_file", CASE_FILES)
def test_feature_pipeline_fixture_column_order_and_checksum(case_file: str) -> None:
    path = FIXTURE_DIR / case_file
    case = json.loads(path.read_text(encoding="utf-8"))
    cols = residual_training_feature_columns()
    row = _build_row(case)

    assert set(row.keys()) >= set(cols)
    for c in cols:
        assert c in row, f"missing column {c}"

    digest = _vector_sha16(row, cols)
    regen = os.environ.get("VINYLIQ_REGEN_FEATURE_FIXTURES", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if regen:
        _regen_fixture_file(path, case, digest)
        pytest.skip(f"regenerated {case_file}")

    expected = case.get("vector_sha16")
    assert expected, f"{case_file} missing vector_sha16"
    assert digest == expected, (
        f"{case['id']}: checksum {digest} != {expected}; "
        "run with VINYLIQ_REGEN_FEATURE_FIXTURES=1 if intentional"
    )


def test_feature_columns_match_vinyliq_schema_length() -> None:
    cols = residual_training_feature_columns()
    assert len(cols) >= 30
    assert cols[0] == "media_grade"
    assert "genre_index" in cols
