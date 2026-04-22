#!/usr/bin/env python3
"""
Ingest Discogs monthly **masters** XML dump into ``masters_features`` in the
feature store SQLite.

Download ``discogs_*_masters.xml.gz`` from https://data.discogs.com/ (CC0),
then::

  PYTHONPATH=. uv run python \\
      price_estimator/scripts/ingest_discogs_masters_dump.py \\
      --dump /path/to/discogs_YYYYMMDD_masters.xml.gz

Use ``--limit`` for a smoke test; ``--dry-run`` to parse only (no DB
writes).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _year_from_master(elem: ET.Element, *, first_text) -> int:
    yt = first_text(elem, "year")
    if yt:
        m = re.match(r"^(\d{4})", yt.strip())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return 0


def master_element_to_row(
    elem: ET.Element,
    *,
    localname,
    first_text,
    all_strings_in_container,
    artists_from_release,
    skip_deleted: bool = True,
) -> dict[str, Any] | None:
    """Map one ``<master>`` element to a ``masters_features`` row dict."""
    if localname(elem.tag) != "master":
        return None
    if skip_deleted and (elem.get("status") or "").strip() == "Deleted":
        return None
    mid = (elem.get("id") or "").strip()
    if not mid or not mid.isdigit():
        return None

    title_raw = first_text(elem, "title")
    title_s = title_raw.strip() if title_raw else ""

    main_rel = first_text(elem, "main_release")
    main_release_id = main_rel.strip() if main_rel else None
    if main_release_id == "":
        main_release_id = None

    year = _year_from_master(elem, first_text=first_text)

    genres_list = all_strings_in_container(elem, "genres", "genre")
    styles_list = all_strings_in_container(elem, "styles", "style")
    primary_genre = genres_list[0] if genres_list else None
    primary_style = styles_list[0] if styles_list else None

    artists = artists_from_release(elem)
    primary_artist_id: str | None = None
    primary_artist_name: str | None = None
    if artists:
        primary_artist_id = artists[0].get("id") or None
        n0 = (artists[0].get("name") or "").strip()
        primary_artist_name = n0 if n0 else None

    dq_raw = first_text(elem, "data_quality")
    data_quality = dq_raw.strip() if dq_raw else None

    return {
        "master_id": mid,
        "main_release_id": main_release_id,
        "title": title_s or None,
        "year": year,
        "primary_artist_id": primary_artist_id,
        "primary_artist_name": primary_artist_name,
        "primary_genre": primary_genre,
        "primary_style": primary_style,
        "artists_json": (
            json.dumps(artists, separators=(",", ":")) if artists else None
        ),
        "genres_json": json.dumps(genres_list, separators=(",", ":"))
        if genres_list
        else None,
        "styles_json": json.dumps(styles_list, separators=(",", ":"))
        if styles_list
        else None,
        "data_quality": data_quality,
    }


def iter_masters_dump_rows(
    path: Path,
    *,
    localname,
    first_text,
    all_strings_in_container,
    artists_from_release,
    open_dump_binary,
    skip_deleted: bool = True,
) -> Iterator[dict[str, Any]]:
    with open_dump_binary(path) as f:
        for _event, elem in ET.iterparse(f, events=("end",)):
            if localname(elem.tag) != "master":
                continue
            row = master_element_to_row(
                elem,
                localname=localname,
                first_text=first_text,
                all_strings_in_container=all_strings_in_container,
                artists_from_release=artists_from_release,
                skip_deleted=skip_deleted,
            )
            if row is not None:
                yield row
            elem.clear()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Discogs masters XML dump into VinylIQ feature store"
        ),
    )
    parser.add_argument(
        "--dump",
        type=Path,
        required=True,
        help="Path to masters.xml or masters.xml.gz",
    )
    parser.add_argument(
        "--feature-db",
        type=Path,
        default=None,
        help=(
            "SQLite for releases_features + masters_features "
            "(default: price_estimator/data/feature_store.sqlite)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many masters (0 = no limit)",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=2000,
        help="Rows per SQLite transaction when writing masters_features",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only; print counts, no writes",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Keep masters with status=Deleted (default: skip)",
    )
    args = parser.parse_args()

    dump_path = args.dump
    if not dump_path.is_file():
        print(f"Dump file not found: {dump_path}", file=sys.stderr)
        return 1

    root = _root()
    skip_deleted = not args.include_deleted

    try:
        from price_estimator.src.ingest.discogs_dump import (
            _all_strings_in_container,
            _artists_from_release,
            _first_text,
            _localname,
            open_dump_binary,
        )
        from price_estimator.src.storage.feature_store import FeatureStoreDB
    except ImportError:
        print("PYTHONPATH must include repo root.", file=sys.stderr)
        return 1

    feat_path = args.feature_db or (root / "data" / "feature_store.sqlite")
    if not feat_path.is_absolute():
        feat_path = root / feat_path

    rows_iter = iter_masters_dump_rows(
        dump_path,
        localname=_localname,
        first_text=_first_text,
        all_strings_in_container=_all_strings_in_container,
        artists_from_release=_artists_from_release,
        open_dump_binary=open_dump_binary,
        skip_deleted=skip_deleted,
    )

    db: FeatureStoreDB | None = None
    if not args.dry_run:
        db = FeatureStoreDB(feat_path)

    batch: list[dict[str, Any]] = []
    total = 0
    try:
        for row in rows_iter:
            if args.limit and total >= args.limit:
                break
            total += 1
            if args.dry_run:
                continue
            assert db is not None
            batch.append(row)
            if len(batch) >= max(1, args.commit_every):
                db.upsert_many_masters(batch)
                batch.clear()
                print(f"... {total} masters", flush=True)
        if batch and db is not None:
            db.upsert_many_masters(batch)
    finally:
        pass

    if args.dry_run:
        print(f"dry-run: parsed {total} masters (no writes)")
    else:
        print(f"done: {total} masters → {feat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
