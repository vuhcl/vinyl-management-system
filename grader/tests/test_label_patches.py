"""Tests for grader.src.data.label_patches."""
from __future__ import annotations

import json
from pathlib import Path

from grader.src.data.label_patches import (
    append_csv_to_label_patches,
    apply_label_patches_after_ingest,
    apply_label_patches_to_processed_file,
    export_label_patches_from_jsonl,
    load_label_patches,
    merge_csv_into_label_patches,
)


def test_load_label_patches_skips_comments_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    p.write_text(
        "# comment\n\n"
        '{"item_id": "1", "source": "discogs", "sleeve_label": "Good"}\n',
        encoding="utf-8",
    )
    rows = load_label_patches(p)
    assert len(rows) == 1
    assert rows[0]["item_id"] == "1"


def test_apply_updates_discogs_rows(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    disc = proc / "discogs_processed.jsonl"
    rec = {
        "item_id": "99",
        "source": "discogs",
        "text": "note",
        "sleeve_label": "Good",
        "media_label": "Good",
        "label_confidence": 1.0,
    }
    disc.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    patch = tmp_path / "patch.jsonl"
    patch.write_text(
        '{"item_id": "99", "source": "discogs", "sleeve_label": "Very Good Plus"}\n',
        encoding="utf-8",
    )
    from grader.src.data.label_patches import _build_patch_index

    patches = load_label_patches(patch)
    index, _ = _build_patch_index(patches)
    st = apply_label_patches_to_processed_file(
        disc, source="discogs", index=index, dry_run=False
    )
    assert st["updated"] == 1
    loaded = json.loads(disc.read_text(encoding="utf-8").strip().split("\n")[0])
    assert loaded["sleeve_label"] == "Very Good Plus"
    assert loaded["media_label"] == "Good"


def test_apply_label_patches_after_ingest_from_config(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "discogs_processed.jsonl").write_text(
        json.dumps(
            {
                "item_id": "a",
                "source": "discogs",
                "text": "x",
                "sleeve_label": "Poor",
                "media_label": "Poor",
                "label_confidence": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    patch = tmp_path / "patch.jsonl"
    patch.write_text(
        '{"item_id": "a", "source": "discogs", "media_label": "Near Mint"}\n',
        encoding="utf-8",
    )
    cfg = {
        "paths": {"processed": str(proc)},
        "data": {"label_patches_path": str(patch)},
    }
    out = apply_label_patches_after_ingest(cfg)
    assert out["updated_total"] == 1
    assert "discogs_sale_history" in out
    row = json.loads((proc / "discogs_processed.jsonl").read_text().strip())
    assert row["media_label"] == "Near Mint"
    assert row["sleeve_label"] == "Poor"


def test_apply_label_patches_sale_history(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    sh = proc / "discogs_sale_history.jsonl"
    rec = {
        "item_id": "sale-1",
        "source": "discogs_sale_history",
        "text": "played once",
        "sleeve_label": "Good",
        "media_label": "Good",
        "label_confidence": 1.0,
    }
    sh.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    patch = tmp_path / "patch.jsonl"
    patch.write_text(
        '{"item_id": "sale-1", "source": "discogs_sale_history", '
        '"sleeve_label": "Very Good Plus"}\n',
        encoding="utf-8",
    )
    from grader.src.data.label_patches import _build_patch_index

    patches = load_label_patches(patch)
    index, _ = _build_patch_index(patches)
    st = apply_label_patches_to_processed_file(
        sh, source="discogs_sale_history", index=index, dry_run=False
    )
    assert st["updated"] == 1
    loaded = json.loads(sh.read_text(encoding="utf-8").strip())
    assert loaded["sleeve_label"] == "Very Good Plus"


def test_append_csv_to_label_patches(tmp_path: Path) -> None:
    csv_p = tmp_path / "in.csv"
    csv_p.write_text(
        "item_id,source,sleeve_label,media_label\n"
        'x1,discogs,Very Good,Near Mint\n'
        'x2,bad_source,Good,Good\n',
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    dest.write_text('{"item_id": "old", "source": "discogs"}\n', encoding="utf-8")
    stats = append_csv_to_label_patches(csv_p, dest, dry_run=False)
    assert stats["appended"] == 1
    assert stats["skipped_rows"] == 1
    lines = dest.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    last = json.loads(lines[-1])
    assert last["item_id"] == "x1"
    assert last["sleeve_label"] == "Very Good"


def test_merge_csv_into_label_patches(tmp_path: Path) -> None:
    dest = tmp_path / "p.jsonl"
    dest.write_text(
        '{"item_id": "1", "source": "discogs", "sleeve_label": "Good", '
        '"media_label": "Mint"}\n',
        encoding="utf-8",
    )
    csv_p = tmp_path / "in.csv"
    csv_p.write_text(
        "item_id,source,sleeve_label,media_label\n"
        "1,discogs,Very Good Plus,\n"
        "2,discogs,Near Mint,Near Mint\n",
        encoding="utf-8",
    )
    st = merge_csv_into_label_patches(csv_p, dest, dry_run=False)
    assert st["appended"] == 1
    assert st["updated_sleeve"] == 1
    assert st["skipped_rows"] == 0
    lines = [json.loads(s) for s in dest.read_text(encoding="utf-8").splitlines() if s]
    assert len(lines) == 2
    u = next(x for x in lines if x["item_id"] == "1")
    a = next(x for x in lines if x["item_id"] == "2")
    assert u["sleeve_label"] == "Very Good Plus" and u["media_label"] == "Mint"
    assert a == {
        "item_id": "2",
        "source": "discogs",
        "sleeve_label": "Near Mint",
        "media_label": "Near Mint",
    }


def test_export_label_patches_from_unified(tmp_path: Path) -> None:
    uni = tmp_path / "unified.jsonl"
    uni.write_text(
        json.dumps(
            {
                "item_id": "10",
                "source": "discogs",
                "text": "a",
                "sleeve_label": "Good",
                "media_label": "Very Good",
            }
        )
        + "\n"
        + json.dumps(
            {
                "item_id": "11",
                "source": "ebay_jp",
                "sleeve_label": "Mint",
                "media_label": "Mint",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outp = tmp_path / "out.jsonl"
    st = export_label_patches_from_jsonl(uni, outp)
    assert st["exported"] == 2
    lines = [json.loads(x) for x in outp.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert lines[0] == {
        "item_id": "10",
        "source": "discogs",
        "sleeve_label": "Good",
        "media_label": "Very Good",
    }


def test_apply_dry_run_does_not_write(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    disc = proc / "discogs_processed.jsonl"
    original = {
        "item_id": "z",
        "source": "discogs",
        "text": "n",
        "sleeve_label": "Mint",
        "media_label": "Mint",
        "label_confidence": 1.0,
    }
    disc.write_text(json.dumps(original) + "\n", encoding="utf-8")
    patch = tmp_path / "patch.jsonl"
    patch.write_text(
        '{"item_id": "z", "source": "discogs", "sleeve_label": "Good"}\n',
        encoding="utf-8",
    )
    cfg = {"paths": {"processed": str(proc)}, "data": {"label_patches_path": str(patch)}}
    apply_label_patches_after_ingest(cfg, dry_run=True)
    row = json.loads(disc.read_text().strip())
    assert row["sleeve_label"] == "Mint"
