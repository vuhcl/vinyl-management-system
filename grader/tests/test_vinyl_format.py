"""Tests for ``grader.src.data.vinyl_format`` (Discogs release.format)."""
from __future__ import annotations

import json

import pytest

from grader.src.data.vinyl_format import release_format_looks_like_physical_vinyl


@pytest.mark.parametrize(
    ("fmt", "desc", "expect"),
    [
        ("(Vinyl, LP, Album)", "", True),
        ("(LP, Album)", "", True),
        ('(12", EP)', "", True),
        ('(8", 33 RPM, EP)', "", True),
        ("(CD, Vinyl, LP, Box Set)", "", True),
        ("(CD, Album)", "", False),
        ("(Cassette, Album)", "", False),
        ("(DVD-Video)", "", False),
        ("Limited Edition", "", False),
    ],
)
def test_release_format_looks_like_physical_vinyl(
    fmt: str,
    desc: str,
    expect: bool,
) -> None:
    assert release_format_looks_like_physical_vinyl(fmt, desc) is expect


def test_filter_discogs_processed_drops_cd_keeps_vinyl(tmp_path: Path) -> None:
    from grader.src.data.vinyl_format import filter_discogs_processed_vinyl_jsonl

    p = tmp_path / "d.jsonl"
    rows: list[dict] = [
        {
            "item_id": "1",
            "source": "discogs",
            "sleeve_label": "Mint",
            "media_label": "Mint",
            "release_format": "(CD, Album)",
            "release_description": "",
        },
        {
            "item_id": "2",
            "source": "discogs",
            "sleeve_label": "Mint",
            "media_label": "Mint",
            "release_format": "(Vinyl, LP, Album)",
            "release_description": "",
        },
        {
            "item_id": "3",
            "source": "ebay_jp",
            "sleeve_label": "Mint",
            "media_label": "Mint",
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    st = filter_discogs_processed_vinyl_jsonl(p, dry_run=False)
    assert st["dropped"] == 1
    assert st["kept"] == 2
    out = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(out) == 2
    assert {r["item_id"] for r in out} == {"2", "3"}


def test_run_post_patch_respects_config_stage(tmp_path: Path) -> None:
    from grader.src.data.vinyl_format import run_post_patch_vinyl_filter_from_config

    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "discogs_processed.jsonl").write_text("", encoding="utf-8")
    cfg_fetch = {
        "paths": {"processed": str(proc)},
        "data": {"discogs": {"vinyl_format_filter_stage": "fetch"}},
    }
    assert run_post_patch_vinyl_filter_from_config(cfg_fetch)["ran"] is False
    cfg_post = {
        "paths": {"processed": str(proc)},
        "data": {
            "discogs": {"vinyl_format_filter_stage": "post_patch"},
            "sale_history": {"run_sale_vinyl_in_post_hook": False},
        },
    }
    assert run_post_patch_vinyl_filter_from_config(cfg_post)["ran"] is True


def test_format_fields_from_releases_features_prefers_format_desc() -> None:
    from grader.src.data.vinyl_format import format_fields_from_releases_features

    r, d = format_fields_from_releases_features('(Vinyl, LP, Album)', "[]")
    assert "vinyl" in r.lower() or "lp" in r.lower()
    assert d == ""


def test_format_fields_from_releases_features_formats_json_list() -> None:
    from grader.src.data.vinyl_format import format_fields_from_releases_features

    raw = json.dumps(
        [
            {
                "name": "Vinyl",
                "qty": "1",
                "descriptions": ["LP", "Album"],
            }
        ]
    )
    r, d = format_fields_from_releases_features("", raw)
    assert "vinyl" in r.lower() and "lp" in r.lower()
    assert d == ""


def test_filter_discogs_sale_history_drops_non_vinyl(tmp_path: Path) -> None:
    from grader.src.data.vinyl_format import (
        DISCOGS_SALE_HISTORY_SOURCE,
        filter_discogs_sale_history_vinyl_jsonl,
    )

    p = tmp_path / "sh.jsonl"
    rows = [
        {
            "item_id": "1:a",
            "source": DISCOGS_SALE_HISTORY_SOURCE,
            "text": "x" * 50,
            "release_format": "(CD, Album)",
            "release_description": "",
        },
        {
            "item_id": "2:b",
            "source": DISCOGS_SALE_HISTORY_SOURCE,
            "text": "x" * 50,
            "release_format": "(Vinyl, LP)",
            "release_description": "",
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    st = filter_discogs_sale_history_vinyl_jsonl(p, dry_run=False)
    assert st["dropped"] == 1
    out = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(out) == 1
    assert out[0]["item_id"] == "2:b"
