"""
Synthetic records for quick benchmark / resume-style metrics.

Mirrors the structure used in ``grader/tests/conftest.py`` so metrics stay
aligned with CI. **Not** a substitute for evaluation on real marketplace data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# Representative seller-note text per sleeve grade (test-fixture style)
GRADE_TEXTS: dict[str, str] = {
    "Mint": "factory sealed, still in shrink wrap, never opened",
    "Near Mint": "never played, no marks whatsoever, barely played once",
    "Excellent": "minor scuff on cover, well cared for, excellent condition",
    "Very Good Plus": "plays perfectly, very light scratch, small seam split",
    "Very Good": "surface noise on quiet passages, light scratches on vinyl",
    "Good": "heavy scratches, crackling throughout, seam split at spine",
    "Poor": "badly warped, skipping on side two, won't play properly",
    "Generic": "generic white sleeve, die-cut inner sleeve only",
}

GRADE_TEXT_VARIANTS: dict[str, list[str]] = {
    "Mint": [
        "still sealed in original shrink",
        "factory sealed, unplayed",
        "sealed copy, mint condition",
        "new and sealed",
        "shrink intact, unplayed",
    ],
    "Near Mint": [
        "one play only, no marks at all",
        "like new, no visible wear",
        "mint minus, barely used",
        "no defects, excellent shape",
        "barely played, pristine",
    ],
    "Excellent": [
        "slight scuff on cover only",
        "very minor wear, carefully handled",
        "light marks on sleeve, plays great",
        "well cared for, minor cosmetic wear",
        "excellent shape, minor blemish",
    ],
    "Very Good Plus": [
        "plays perfectly, minor cosmetic wear only",
        "light scuff on cover, sounds great",
        "very light scratch, plays fine",
        "cosmetic wear only, no audio issues",
        "plays well, turned up corners",
    ],
    "Very Good": [
        "some surface noise, visible scratches",
        "audible noise on quiet passages",
        "plays with some noise, worn",
        "groove wear evident, noisy",
        "light scratches affect sound",
    ],
    "Good": [
        "significant crackling, seam split",
        "heavy wear throughout, tape on cover",
        "lots of surface noise, crackle",
        "writing on label, heavy scratches",
        "plays through, heavy wear",
    ],
    "Poor": [
        "skipping repeatedly, unplayable",
        "cracked, badly warped record",
        "won't play through without skipping",
        "heavily damaged, deep gouges",
        "groove damage, won't play",
    ],
    "Generic": [
        "plain white sleeve, no original cover",
        "company sleeve only, no original",
        "promo copy, generic sleeve",
        "die cut sleeve, missing original cover",
        "blank sleeve, no artwork",
    ],
}


def load_canonical_grades(
    guidelines_path: str | Path,
) -> tuple[list[str], list[str]]:
    path = Path(guidelines_path)
    with open(path, encoding="utf-8") as f:
        g = yaml.safe_load(f)
    return list(g["sleeve_grades"]), list(g["media_grades"])


def build_synthetic_unified_records(
    guidelines_path: str | Path,
) -> list[dict[str, Any]]:
    """
    One primary text + five variants per sleeve grade × two sources
    (same recipe as ``sample_unified_records`` in conftest).
    """
    sleeve_grades, media_grades = load_canonical_grades(guidelines_path)
    media_set = set(media_grades)
    records: list[dict[str, Any]] = []
    record_id = 0
    for source in ("discogs", "ebay_jp"):
        for sleeve_grade in sleeve_grades:
            if sleeve_grade == "Generic":
                media_grade = "Near Mint"
            elif sleeve_grade in media_set:
                media_grade = sleeve_grade
            else:
                media_grade = "Very Good Plus"
            texts = [GRADE_TEXTS[sleeve_grade]] + GRADE_TEXT_VARIANTS[
                sleeve_grade
            ]
            for text in texts:
                records.append(
                    {
                        "item_id": str(record_id),
                        "source": source,
                        "text": text,
                        "sleeve_label": sleeve_grade,
                        "media_label": media_grade,
                        "label_confidence": (
                            1.0 if source == "discogs" else 0.90
                        ),
                        "media_verifiable": sleeve_grade != "Mint",
                        "obi_condition": "VG+" if source == "ebay_jp" else None,
                        "raw_sleeve": sleeve_grade,
                        "raw_media": media_grade,
                        "artist": "Test Artist",
                        "title": f"Test Album {record_id}",
                        "year": 1975,
                        "country": "JP" if source == "ebay_jp" else "US",
                    }
                )
                record_id += 1
    return records


def write_unified_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
