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


# Source string for grader sale-history JSONL (must match ingest_sale_history).
DISCOGS_SALE_HISTORY_SOURCE = "discogs_sale_history"


def format_fields_from_releases_features(
    format_desc: str | None, formats_json: str | None
) -> tuple[str, str]:
    """
    Map ``releases_features`` columns to ``(release_format, release_description)``.

    ``release_format_looks_like_physical_vinyl`` applies non-vinyl substrings to the
    first string only, so the catalog text (e.g. ``cd``, ``vinyl``) should live in
    ``release_format`` when possible.
    """
    fd = (format_desc or "").strip()
    if fd:
        return fd, ""
    raw = (formats_json or "").strip()
    if not raw:
        return "", ""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:2000], ""
    if isinstance(data, list):
        parts: list[str] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            name = (it.get("name") or "").strip()
            descs = it.get("descriptions")
            if isinstance(descs, list):
                d = " ".join(str(x) for x in descs)
            else:
                d = str(descs or "").strip()
            chunk = f"{name} {d}".strip()
            if chunk:
                parts.append(chunk)
        merged = " ".join(parts).strip()
        if merged:
            return merged, ""
    return "", ""


def filter_records_vinyl_by_source(
    records: list[dict[str, Any]],
    *,
    source_allowlist: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """
    Keep rows that pass the vinyl test for allowlisted ``source`` values.

    For allowlisted sources: if both ``release_format`` and ``release_description``
    are empty, keep (legacy / unknown). Otherwise keep iff
    ``release_format_looks_like_physical_vinyl`` is True.
    For other sources, keep all (e.g. eBay in ``discogs_processed``).
    """
    kept: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        src = str(rec.get("source") or "")
        if src not in source_allowlist:
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
    return kept, dropped


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
    ingests). eBay rows and other non-``discogs`` sources are always kept.
    """
    if not jsonl_path.is_file():
        return {"path": str(jsonl_path), "exists": False, "dropped": 0, "kept": 0}

    records = _load_jsonl(jsonl_path)
    kept, dropped = filter_records_vinyl_by_source(
        records, source_allowlist={"discogs"}
    )

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


def filter_discogs_sale_history_vinyl_jsonl(
    jsonl_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Filter ``discogs_sale_history`` JSONL the same way as allowlisted discogs
    sources (empty format fields → keep; else vinyl-like only).
    """
    if not jsonl_path.is_file():
        return {"path": str(jsonl_path), "exists": False, "dropped": 0, "kept": 0}

    records = _load_jsonl(jsonl_path)
    kept, dropped = filter_records_vinyl_by_source(
        records, source_allowlist={DISCOGS_SALE_HISTORY_SOURCE}
    )

    if not dry_run and (dropped or len(kept) != len(records)):
        _write_jsonl(jsonl_path, kept)
        logger.info(
            "post_patch sale_history vinyl filter: dropped %d row(s); %d in %s",
            dropped,
            len(kept),
            jsonl_path,
        )
    elif dry_run:
        logger.info(
            "post_patch sale_history vinyl filter (dry-run): would drop %d row(s)",
            dropped,
        )

    return {
        "path": str(jsonl_path),
        "exists": True,
        "dropped": dropped,
        "kept": len(kept),
        "dry_run": dry_run,
    }


def _sale_history_jsonl_path_from_config(config: dict[str, Any]) -> Path:
    sh_cfg = (config.get("data") or {}).get("sale_history") or {}
    processed = Path(config["paths"]["processed"])
    if not processed.is_absolute():
        processed = Path.cwd() / processed
    raw_sh = sh_cfg.get("processed_jsonl")
    if raw_sh:
        sh_path = Path(raw_sh)
    else:
        sh_path = processed / "discogs_sale_history.jsonl"
    if not sh_path.is_absolute():
        sh_path = Path.cwd() / sh_path
    return sh_path


def run_post_patch_vinyl_filter_from_config(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    filter_sale_jsonl: bool | None = None,
) -> dict[str, Any]:
    """
    If ``vinyl_format_filter_stage`` is ``post_patch``, filter ``discogs_processed`` JSONL.

    When ``data.sale_history.run_sale_vinyl_in_post_hook`` is true, also re-filter
    ``discogs_sale_history`` on disk. Pass ``filter_sale_jsonl=False`` when a sale
    ingest in the same process will write that file (avoids redundant I/O).
    """
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

    sh_cfg = (config.get("data") or {}).get("sale_history") or {}
    if filter_sale_jsonl is None:
        filter_sale_jsonl = bool(
            sh_cfg.get("run_sale_vinyl_in_post_hook", True)
        )

    discogs_path = processed / "discogs_processed.jsonl"
    d_stats = filter_discogs_processed_vinyl_jsonl(discogs_path, dry_run=dry_run)

    out: dict[str, Any] = {
        "ran": True,
        "stage": stage,
        **d_stats,
    }
    if filter_sale_jsonl:
        sh_path = _sale_history_jsonl_path_from_config(config)
        sh_stats = filter_discogs_sale_history_vinyl_jsonl(
            sh_path, dry_run=dry_run
        )
        out["sale_history"] = sh_stats
    return out
