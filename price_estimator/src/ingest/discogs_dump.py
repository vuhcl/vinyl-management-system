"""
Stream-parse Discogs monthly `releases` XML (.xml or .xml.gz) into feature rows.

Dump schema follows Discogs' public XML export (see data.discogs.com). Tag names
are matched on the local part (no-namespace dumps).
"""
from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

from .release_row import release_row_from_fields
from ..features.vinyliq_features import (
    format_flags_from_text,
    is_original_pressing_from_formats_list,
)


def _localname(tag: str) -> str:
    if tag and tag[0] == "{":
        return tag.rsplit("}", 1)[-1]
    return tag


def _first_text(parent: ET.Element, name: str) -> str | None:
    for c in parent:
        if _localname(c.tag) == name:
            t = (c.text or "").strip()
            return t if t else None
    return None


def _all_strings_in_container(
    parent: ET.Element,
    container: str,
    item: str,
) -> list[str]:
    out: list[str] = []
    for c in parent:
        if _localname(c.tag) != container:
            continue
        for sub in c:
            if _localname(sub.tag) == item and sub.text:
                t = sub.text.strip()
                if t:
                    out.append(t)
    return out


def _artists_from_release(elem: ET.Element) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for c in elem:
        if _localname(c.tag) != "artists":
            continue
        for a in c:
            if _localname(a.tag) != "artist":
                continue
            aid = _first_text(a, "id")
            name = _first_text(a, "name")
            if aid and aid.strip().isdigit():
                out.append(
                    {"id": aid.strip(), "name": (name or "").strip()},
                )
    return out


def _labels_from_release(elem: ET.Element) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for c in elem:
        if _localname(c.tag) != "labels":
            continue
        for lab in c:
            if _localname(lab.tag) != "label":
                continue
            lid = (lab.get("id") or "").strip()
            name = (lab.get("name") or "").strip()
            catno = (lab.get("catno") or "").strip()
            if not lid:
                tid = _first_text(lab, "id")
                lid = tid.strip() if tid else ""
            if not name:
                tname = _first_text(lab, "name")
                name = tname.strip() if tname else ""
            if not catno:
                tcat = _first_text(lab, "catno")
                catno = tcat.strip() if tcat else ""
            if lid or name or catno:
                out.append({"id": lid, "name": name, "catno": catno})
    return out


def _formats_from_release(elem: ET.Element) -> tuple[list[dict[str, Any]], str | None]:
    fmts: list[dict[str, Any]] = []
    desc_parts: list[str] = []
    for c in elem:
        if _localname(c.tag) != "formats":
            continue
        for fmt in c:
            if _localname(fmt.tag) != "format":
                continue
            name = (fmt.get("name") or "").strip()
            qty = (fmt.get("qty") or "").strip()
            descs: list[str] = []
            for ch in fmt:
                if _localname(ch.tag) != "descriptions":
                    continue
                for d in ch:
                    if _localname(d.tag) == "description" and d.text:
                        t = d.text.strip()
                        if t:
                            descs.append(t)
            fmts.append({"name": name, "qty": qty, "descriptions": descs})
            extra = " ".join(descs)
            desc_parts.append(f"{name} {extra}".strip())
    format_desc = ", ".join(p for p in desc_parts if p) or None
    return fmts, format_desc


def _year_from_release(elem: ET.Element) -> int:
    yt = _first_text(elem, "year")
    if yt:
        m = re.match(r"^(\d{4})", yt.strip())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    rel = _first_text(elem, "released")
    if rel:
        m = re.match(r"^(\d{4})", rel.strip())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return 0


def _community_counts(elem: ET.Element) -> tuple[int, int]:
    for c in elem:
        if _localname(c.tag) != "community":
            continue
        want = 0
        have = 0
        for sub in c:
            ln = _localname(sub.tag)
            if ln == "want" and sub.text:
                try:
                    want = int(sub.text.strip())
                except ValueError:
                    want = 0
            elif ln == "have" and sub.text:
                try:
                    have = int(sub.text.strip())
                except ValueError:
                    have = 0
        return want, have
    return 0, 0


def release_element_to_row(
    elem: ET.Element,
    *,
    skip_deleted: bool = True,
) -> dict[str, Any] | None:
    """
    Map one ``<release>`` element to a ``FeatureStoreDB`` row dict.
    Returns None if the element should be skipped (no id, deleted, etc.).
    """
    if _localname(elem.tag) != "release":
        return None
    if skip_deleted and (elem.get("status") or "").strip() == "Deleted":
        return None
    rid = (elem.get("id") or "").strip()
    if not rid or not rid.isdigit():
        return None

    master_raw = _first_text(elem, "master_id")
    master_id_s = master_raw.strip() if master_raw else None
    if master_id_s == "":
        master_id_s = None

    genres_list = _all_strings_in_container(elem, "genres", "genre")
    styles_list = _all_strings_in_container(elem, "styles", "style")
    genre = genres_list[0] if genres_list else None
    style = styles_list[0] if styles_list else None

    artists = _artists_from_release(elem)
    labels = _labels_from_release(elem)
    formats_list, format_desc = _formats_from_release(elem)

    year = _year_from_release(elem)
    decade = (year // 10) * 10 if year else 0

    country_raw = _first_text(elem, "country")
    country_s = country_raw.strip() if country_raw else None

    flags = format_flags_from_text(format_desc)
    is_original = is_original_pressing_from_formats_list(formats_list)

    return release_row_from_fields(
        release_id=rid,
        master_id=master_id_s,
        genre=genre,
        style=style,
        decade=decade,
        year=year,
        country=country_s,
        label_tier=0,
        is_original_pressing=is_original,
        is_colored_vinyl=flags["is_colored_vinyl"],
        is_picture_disc=flags["is_picture_disc"],
        is_promo=flags["is_promo"],
        format_desc=format_desc,
        artists_json=json.dumps(artists, separators=(",", ":")) if artists else None,
        labels_json=json.dumps(labels, separators=(",", ":")) if labels else None,
        genres_json=json.dumps(genres_list, separators=(",", ":")) if genres_list else None,
        styles_json=json.dumps(styles_list, separators=(",", ":")) if styles_list else None,
        formats_json=json.dumps(formats_list, separators=(",", ":")) if formats_list else None,
    )


def open_dump_binary(path: Path):
    """Open a dump file for reading (gzip if path ends with .gz)."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def iter_release_elements(path: Path) -> Iterator[ET.Element]:
    """
    Stream ``<release>...</release>`` elements; clears each element after yield
    to bound memory (stdlib ElementTree; very deep trees may still use RAM).
    """
    with open_dump_binary(path) as f:
        for _event, elem in ET.iterparse(f, events=("end",)):
            if _localname(elem.tag) == "release":
                yield elem
                elem.clear()


def iter_dump_feature_rows(
    path: Path,
    *,
    skip_deleted: bool = True,
) -> Iterator[dict[str, Any]]:
    """Parse *path* and yield feature-store row dicts."""
    for elem in iter_release_elements(path):
        row = release_element_to_row(elem, skip_deleted=skip_deleted)
        if row is not None:
            yield row


def probe_dump_community(
    path: Path,
    *,
    limit: int,
    skip_deleted: bool = True,
) -> tuple[int, int, int]:
    """
    Scan the first *limit* accepted releases and summarize community stats.

    Returns ``(parsed_rows, count_with_have_or_want_positive,
    max_have_plus_want)``.

    If ``count_with_have_or_want_positive`` is 0, the dump likely omits
    ``<community><have>`` / ``<want>`` (common on recent public dumps);
    popularity ordering in SQLite will fall back to ``release_id``.
    """
    parsed = 0
    nz = 0
    max_sum = 0
    for elem in iter_release_elements(path):
        row = release_element_to_row(elem, skip_deleted=skip_deleted)
        if row is None:
            continue
        parsed += 1
        w, h = _community_counts(elem)
        s = int(w) + int(h)
        if s > 0:
            nz += 1
        if s > max_sum:
            max_sum = s
        if parsed >= limit:
            break
    return parsed, nz, max_sum
