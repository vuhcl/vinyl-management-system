"""Tests for grader.src.eval.generate_relabel_review_csv."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from grader.src.eval import generate_relabel_review_csv as mod


def _write_cleanlab(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_patches(path: Path, objs: list[dict[str, str]]) -> None:
    import json

    path.write_text(
        "\n".join(json.dumps(o) for o in objs) + "\n", encoding="utf-8"
    )


def _write_candidates(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_suggest_media_old_oof_branch(tmp_path: Path) -> None:
    """(M,O) pair has >=2 cohort rows -> cohort_media_old_oof."""
    patches = tmp_path / "patches.jsonl"
    cl = tmp_path / "cleanlab.csv"
    cand = tmp_path / "cand.csv"
    out = tmp_path / "out.csv"

    _write_cleanlab(
        cl,
        [
            {
                "item_id": "a1",
                "source": "s",
                "media_label": "Good",
                "sleeve_label": "VG",
                "oof_pred_label": "Very Good",
            },
            {
                "item_id": "a2",
                "source": "s",
                "media_label": "Good",
                "sleeve_label": "VG",
                "oof_pred_label": "Very Good",
            },
        ],
    )
    _write_patches(
        patches,
        [
            {
                "item_id": "a1",
                "source": "s",
                "media_label": "Near Mint",
                "sleeve_label": "NM",
            },
            {
                "item_id": "a2",
                "source": "s",
                "media_label": "Near Mint",
                "sleeve_label": "NM",
            },
        ],
    )
    _write_candidates(
        cand,
        [
            {
                "item_id": "x",
                "source": "s",
                "media_label_at_audit": "Good",
                "oof_pred_label": "Very Good",
                "sleeve_label": "VG",
                "cleanlab_self_confidence": "0.1",
                "text_pattern_score": "1",
                "text_token_hits": "a b",
                "modeling_text_snippet": "x" * 300,
            },
        ],
    )

    mod.write_review_csv(
        patches_path=patches,
        patches_tail=10,
        cleanlab_media_path=cl,
        candidates_csv=cand,
        output_path=out,
        snippet_max=240,
    )
    with out.open(encoding="utf-8", newline="") as f:
        r = list(csv.DictReader(f))[0]
    assert r["suggested_media_label"] == "Near Mint"
    assert r["suggestion_source_media"] == "cohort_media_old_oof"
    assert len(r["modeling_text_snippet"]) == 240


def test_suggest_media_old_only_branch(tmp_path: Path) -> None:
    """Marginal old_media only >=3 -> cohort_media_old_only."""
    patches = tmp_path / "patches.jsonl"
    cl = tmp_path / "cleanlab.csv"
    cand = tmp_path / "cand.csv"
    out = tmp_path / "out.csv"

    _write_cleanlab(
        cl,
        [
            {
                "item_id": f"id{i}",
                "source": "s",
                "media_label": "Fair",
                "sleeve_label": "G",
                "oof_pred_label": f"O{i}",
            }
            for i in range(3)
        ],
    )
    _write_patches(
        patches,
        [
            {
                "item_id": f"id{i}",
                "source": "s",
                "media_label": "Good Plus",
                "sleeve_label": "VG+",
            }
            for i in range(3)
        ],
    )
    _write_candidates(
        cand,
        [
            {
                "item_id": "cand",
                "source": "s",
                "media_label_at_audit": "Fair",
                "oof_pred_label": "O9",
                "sleeve_label": "G",
                "cleanlab_self_confidence": "",
                "text_pattern_score": "",
                "text_token_hits": "",
                "modeling_text_snippet": "",
            },
        ],
    )

    mod.write_review_csv(
        patches_path=patches,
        patches_tail=10,
        cleanlab_media_path=cl,
        candidates_csv=cand,
        output_path=out,
    )
    with out.open(encoding="utf-8", newline="") as f:
        r = list(csv.DictReader(f))[0]
    assert r["suggested_media_label"] == "Good Plus"
    assert r["suggestion_source_media"] == "cohort_media_old_only"


def test_suggest_media_oof_fallback(tmp_path: Path) -> None:
    patches = tmp_path / "patches.jsonl"
    cl = tmp_path / "cleanlab.csv"
    cand = tmp_path / "cand.csv"
    out = tmp_path / "out.csv"

    _write_cleanlab(
        cl,
        [
            {
                "item_id": "only",
                "source": "s",
                "media_label": "Poor",
                "sleeve_label": "P",
                "oof_pred_label": "Good",
            },
        ],
    )
    _write_patches(
        patches,
        [
            {
                "item_id": "only",
                "source": "s",
                "media_label": "Fair",
                "sleeve_label": "F",
            },
        ],
    )
    _write_candidates(
        cand,
        [
            {
                "item_id": "c",
                "source": "s",
                "media_label_at_audit": "Poor",
                "oof_pred_label": "Very Good Plus",
                "sleeve_label": "P",
                "cleanlab_self_confidence": "",
                "text_pattern_score": "",
                "text_token_hits": "",
                "modeling_text_snippet": "",
            },
        ],
    )

    mod.write_review_csv(
        patches_path=patches,
        patches_tail=10,
        cleanlab_media_path=cl,
        candidates_csv=cand,
        output_path=out,
    )
    with out.open(encoding="utf-8", newline="") as f:
        r = list(csv.DictReader(f))[0]
    assert r["suggested_media_label"] == "Very Good Plus"
    assert r["suggestion_source_media"] == "oof_pred_fallback"


def test_suggest_sleeve_blank_without_cohort(tmp_path: Path) -> None:
    patches = tmp_path / "patches.jsonl"
    cl = tmp_path / "cleanlab.csv"
    cand = tmp_path / "cand.csv"
    out = tmp_path / "out.csv"

    _write_cleanlab(
        cl,
        [
            {
                "item_id": "u",
                "source": "s",
                "media_label": "M",
                "sleeve_label": "UniqueSleeve",
                "oof_pred_label": "M",
            },
        ],
    )
    _write_patches(
        patches,
        [
            {
                "item_id": "u",
                "source": "s",
                "media_label": "M",
                "sleeve_label": "X",
            },
        ],
    )
    _write_candidates(
        cand,
        [
            {
                "item_id": "c",
                "source": "s",
                "media_label_at_audit": "M",
                "oof_pred_label": "M",
                "sleeve_label": "UniqueSleeve",
                "cleanlab_self_confidence": "",
                "text_pattern_score": "",
                "text_token_hits": "",
                "modeling_text_snippet": "",
            },
        ],
    )

    mod.write_review_csv(
        patches_path=patches,
        patches_tail=10,
        cleanlab_media_path=cl,
        candidates_csv=cand,
        output_path=out,
    )
    with out.open(encoding="utf-8", newline="") as f:
        r = list(csv.DictReader(f))[0]
    assert r["suggested_sleeve_label"] == ""


def test_suggest_sleeve_cohort(tmp_path: Path) -> None:
    patches = tmp_path / "patches.jsonl"
    cl = tmp_path / "cleanlab.csv"
    cand = tmp_path / "cand.csv"
    out = tmp_path / "out.csv"

    _write_cleanlab(
        cl,
        [
            {
                "item_id": "s1",
                "source": "s",
                "media_label": "A",
                "sleeve_label": "OldS",
                "oof_pred_label": "A",
            },
            {
                "item_id": "s2",
                "source": "s",
                "media_label": "A",
                "sleeve_label": "OldS",
                "oof_pred_label": "A",
            },
        ],
    )
    _write_patches(
        patches,
        [
            {
                "item_id": "s1",
                "source": "s",
                "media_label": "A",
                "sleeve_label": "NewS",
            },
            {
                "item_id": "s2",
                "source": "s",
                "media_label": "A",
                "sleeve_label": "NewS",
            },
        ],
    )
    _write_candidates(
        cand,
        [
            {
                "item_id": "c",
                "source": "s",
                "media_label_at_audit": "A",
                "oof_pred_label": "A",
                "sleeve_label": "OldS",
                "cleanlab_self_confidence": "",
                "text_pattern_score": "",
                "text_token_hits": "",
                "modeling_text_snippet": "",
            },
        ],
    )

    mod.write_review_csv(
        patches_path=patches,
        patches_tail=10,
        cleanlab_media_path=cl,
        candidates_csv=cand,
        output_path=out,
    )
    with out.open(encoding="utf-8", newline="") as f:
        r = list(csv.DictReader(f))[0]
    assert r["suggested_sleeve_label"] == "NewS"


def test_main_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.csv"
    code = mod.main(
        [
            "--patches-path",
            str(tmp_path / "no_p.jsonl"),
            "--cleanlab-media",
            str(missing),
            "--candidates-csv",
            str(missing),
            "--output",
            str(tmp_path / "o.csv"),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "Missing" in err
