"""
Physical-vinyl detection for Discogs inventory ``release.format`` / description.

Used by ``ingest_discogs`` when ``data.discogs.format_filter`` is Vinyl-like.
Multi-format listings (e.g. CD + vinyl box sets) count if **any** vinyl signal
appears. CD / DVD / cassette / digital-only lines are excluded.

When ``data.discogs.vinyl_format_filter_stage`` is ``post_patch``, rows are not
dropped at fetch time; instead ``filter_discogs_processed_vinyl_jsonl`` runs
after label patches (see ``pipeline.train``).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Substrings of ``release.format`` (lowercased) that indicate non-vinyl media.
# Used only after vinyl-positive checks fail (so ``cd, vinyl`` still counts).
_NON_VINYL_FORMAT_SUBSTRINGS: tuple[str, ...] = (
    "cd",
    "dvd",
    "sacd",
    "blu-ray",
    "cassette",
    "tape",
    "digital",
    "mp3",
    "file",
    "flac",
    "alac",
    "aac",
    "download",
    "stream",
)

_INCH_MARKERS: tuple[str, ...] = (
    '3"',
    '6"',
    '7"',
    '8"',
    '10"',
    '12"',
)
_INCH_WORDS: tuple[str, ...] = (
    "3 inch",
    "6 inch",
    "7 inch",
    "8 inch",
    "10 inch",
    "12 inch",
)


def _has_inch_size(blob: str) -> bool:
    b = blob.lower()
    if any(m in b for m in _INCH_MARKERS):
        return True
    return any(w in b for w in _INCH_WORDS)


def release_format_looks_like_physical_vinyl(
    release_format: str,
    release_description: str = "",
) -> bool:
    """
    Return True if Discogs release metadata suggests physical vinyl is included.

    Vinyl-positive signals (any is enough): literal ``vinyl``, word ``lp``,
    or common disc sizes (3\"–12\") including ``N inch`` spellings.
    """
    rf = (release_format or "").strip().lower()
    desc = (release_description or "").strip().lower()
    blob = f"{rf} {desc}".strip()

    if "vinyl" in blob:
        return True

    if re.search(r"\blp\b", blob):
        return True

    if _has_inch_size(blob):
        return True

    if any(tok in rf for tok in _NON_VINYL_FORMAT_SUBSTRINGS):
        return False

    return False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def filter_discogs_processed_vinyl_jsonl(
    jsonl_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Keep only Discogs rows that pass ``release_format_looks_like_physical_vinyl``.

    Rows with no ``release_format`` / ``release_description`` are kept (legacy
    ingests). eBay rows are always kept.
    """
    if not jsonl_path.is_file():
        return {"path": str(jsonl_path), "exists": False, "dropped": 0, "kept": 0}

    records = _load_jsonl(jsonl_path)
    kept: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        if rec.get("source") != "discogs":
            kept.append(rec)
            continue
        rf = str(rec.get("release_format") or "").strip()
        rd = str(rec.get("release_description") or "").strip()
        if not rf and not rd:
            kept.append(rec)
            continue
        if release_format_looks_like_physical_vinyl(rf, rd):
            kept.append(rec)
        else:
            dropped += 1

    if not dry_run and (dropped or len(kept) != len(records)):
        _write_jsonl(jsonl_path, kept)
        logger.info(
            "post_patch vinyl filter: dropped %d discogs row(s); %d line(s) in %s",
            dropped,
            len(kept),
            jsonl_path,
        )
    elif dry_run:
        logger.info(
            "post_patch vinyl filter (dry-run): would drop %d discogs row(s)",
            dropped,
        )

    return {
        "path": str(jsonl_path),
        "exists": True,
        "dropped": dropped,
        "kept": len(kept),
        "dry_run": dry_run,
    }


def run_post_patch_vinyl_filter_from_config(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """If ``vinyl_format_filter_stage`` is ``post_patch``, filter Discogs JSONL."""
    stage = str(
        (config.get("data") or {}).get("discogs", {}).get(
            "vinyl_format_filter_stage", "fetch"
        )
    ).strip().lower()
    if stage != "post_patch":
        return {"ran": False, "stage": stage}
    processed = Path(config["paths"]["processed"])
    if not processed.is_absolute():
        processed = Path.cwd() / processed
    discogs_path = processed / "discogs_processed.jsonl"
    stats = filter_discogs_processed_vinyl_jsonl(discogs_path, dry_run=dry_run)
    return {"ran": True, "stage": stage, **stats}
