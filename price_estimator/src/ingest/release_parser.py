"""Map Discogs GET /releases/{id} (+ optional master) JSON to feature-store row."""
from __future__ import annotations

import json
from typing import Any

from ..features.vinyliq_features import format_flags_from_text


def _fmt_descriptions(formats: list[dict[str, Any]] | None) -> str | None:
    if not formats:
        return None
    parts: list[str] = []
    for f in formats:
        name = (f.get("name") or "").strip()
        desc = f.get("descriptions") or []
        extra = " ".join(str(d) for d in desc) if desc else ""
        parts.append(f"{name} {extra}".strip())
    out = ", ".join(p for p in parts if p)
    return out or None


def _api_artists(release: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for a in release.get("artists") or []:
        if not isinstance(a, dict):
            continue
        aid = a.get("id")
        if aid is None:
            continue
        sid = str(aid).strip()
        if not sid.isdigit():
            continue
        out.append({"id": sid, "name": str(a.get("name") or "").strip()})
    return out


def _api_labels(release: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for lab in release.get("labels") or []:
        if not isinstance(lab, dict):
            continue
        lid = str(lab.get("id") or "").strip()
        name = str(lab.get("name") or "").strip()
        catno = str(lab.get("catno") or "").strip()
        if lid or name or catno:
            out.append({"id": lid, "name": name, "catno": catno})
    return out


def _formats_json_from_api(
    formats: list[dict[str, Any]] | None,
) -> str | None:
    if not formats:
        return None
    fmts: list[dict[str, Any]] = []
    for f in formats:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name") or "").strip()
        qty = str(f.get("qty") or "").strip()
        descs = f.get("descriptions") or []
        desc_list = [str(d).strip() for d in descs if str(d).strip()]
        fmts.append({"name": name, "qty": qty, "descriptions": desc_list})
    return json.dumps(fmts, separators=(",", ":")) if fmts else None


def release_to_feature_row(
    release: dict[str, Any],
    *,
    master: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a row for ``FeatureStoreDB`` from Discogs API release JSON.

    If *master* is provided (GET /masters/{id}), sets ``is_original_pressing`` when
    this release is the catalog's ``main_release``.
    """
    rid = str(release.get("id", "")).strip()
    master_id = release.get("master_id")
    master_id_s = str(master_id).strip() if master_id is not None else None

    comm = release.get("community") or {}
    wants = int(comm.get("want") or 0)
    haves = int(comm.get("have") or 0)
    ratio = (wants / haves) if haves > 0 else 0.0

    genres = [str(g).strip() for g in (release.get("genres") or []) if str(g).strip()]
    styles = [str(s).strip() for s in (release.get("styles") or []) if str(s).strip()]
    genre = genres[0] if genres else None
    style = styles[0] if styles else None

    year_raw = release.get("year")
    try:
        year = int(year_raw) if year_raw is not None and str(year_raw).strip() else 0
    except (TypeError, ValueError):
        year = 0
    decade = (year // 10) * 10 if year else 0

    country = release.get("country")
    country_s = str(country).strip() if country else None

    formats_raw = release.get("formats")
    formats_list = formats_raw if isinstance(formats_raw, list) else None
    format_desc = _fmt_descriptions(formats_list)
    flags = format_flags_from_text(format_desc)

    artists = _api_artists(release)
    labels = _api_labels(release)
    formats_json = _formats_json_from_api(formats_list)

    is_original = 0
    if master is not None and rid:
        try:
            main_rel = master.get("main_release")
            if main_rel is not None and int(main_rel) == int(release.get("id")):
                is_original = 1
        except (TypeError, ValueError):
            pass

    return {
        "release_id": rid,
        "master_id": master_id_s,
        "want_count": wants,
        "have_count": haves,
        "want_have_ratio": ratio,
        "genre": genre,
        "style": style,
        "decade": decade,
        "year": year,
        "country": country_s,
        "label_tier": 0,
        "is_original_pressing": is_original,
        "is_colored_vinyl": flags["is_colored_vinyl"],
        "is_picture_disc": flags["is_picture_disc"],
        "is_promo": flags["is_promo"],
        "format_desc": format_desc,
        "artists_json": json.dumps(artists, separators=(",", ":")) if artists else None,
        "labels_json": json.dumps(labels, separators=(",", ":")) if labels else None,
        "genres_json": json.dumps(genres, separators=(",", ":")) if genres else None,
        "styles_json": json.dumps(styles, separators=(",", ":")) if styles else None,
        "formats_json": formats_json,
    }
