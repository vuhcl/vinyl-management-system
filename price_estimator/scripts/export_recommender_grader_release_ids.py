#!/usr/bin/env python3
"""
Collect unique Discogs ``release_id`` values from:

1. **Recommender** — your collection and wantlist (Discogs API using the same
   rules as ``recommender.pipeline`` / ``ingest_all``, or CSV fallback).
2. **Grader** — Discogs training listings: release ids are read from cached
   seller inventory JSON pages (``.../inventory/<seller>/per_*/page_*.json``).
   Processed JSONL (``discogs_processed.jsonl``, split files) does not store
   release ids in this repo; eBay rows have no Discogs release id.

Loads repo-root ``.env`` (``DISCOGS_USER_TOKEN`` / ``DISCOGS_TOKEN``).

Examples::

  PYTHONPATH=. python price_estimator/scripts/export_recommender_grader_release_ids.py

  PYTHONPATH=. python price_estimator/scripts/export_recommender_grader_release_ids.py \\
      --recommender-config recommender/configs/base.yaml \\
      --data-dir data/raw \\
      --grader-config grader/configs/grader.yaml \\
      --out price_estimator/data/raw/recommender_grader_release_ids.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (repo / path).resolve()


def _load_recommender_usernames_and_data_dir(
    repo: Path, config_path: Path
) -> tuple[list[str], Path | None]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    discogs = cfg.get("discogs") or {}
    usernames = list(discogs.get("usernames") or [])
    # Pipeline default data_dir when not overridden
    data_dir = cfg.get("data_dir")
    if data_dir is not None:
        return usernames, _resolve_path(repo, data_dir)
    return usernames, None


def _collect_recommender_ids(
    *,
    usernames: list[str],
    data_dir: Path | None,
    collection_csv: Path | None,
    wantlist_csv: Path | None,
) -> set[str]:
    try:
        from shared.project_env import load_project_dotenv

        load_project_dotenv()
    except ImportError:
        print("PYTHONPATH must include repo root (shared.project_env).", file=sys.stderr)
        raise SystemExit(1)

    from recommender.src.data.ingest import load_collection, load_wantlist
    from shared.discogs_api.client import personal_access_token_from_env
    from shared.discogs_api import get_user_collection, get_user_wantlist

    ids: set[str] = set()
    token = personal_access_token_from_env()
    got_recommender_from_api = False

    if usernames and token:
        last_api_error: BaseException | None = None
        max_attempts = 8
        for attempt in range(max_attempts):
            batch: set[str] = set()
            try:
                for username in usernames:
                    for df in (
                        get_user_collection(username, user_token=token),
                        get_user_wantlist(username, user_token=token),
                    ):
                        if df is None or df.empty or "album_id" not in df.columns:
                            continue
                        for x in df["album_id"].astype(str):
                            s = x.strip()
                            if s.isdigit():
                                batch.add(s)
                ids.update(batch)
                got_recommender_from_api = True
                break
            except requests.RequestException as exc:
                last_api_error = exc
                if attempt + 1 >= max_attempts:
                    print(
                        "Discogs API error while fetching collection/wantlist; "
                        "falling back to CSV if --data-dir (or config data_dir) has files. "
                        f"({exc.__class__.__name__})",
                        file=sys.stderr,
                    )
                    break
                wait = min(90.0, 5.0 * (1.6**attempt))
                time.sleep(wait)

    if not got_recommender_from_api:
        col = load_collection(
            path=collection_csv,
            data_dir=data_dir,
            from_discogs=False,
        )
        wl = load_wantlist(
            path=wantlist_csv,
            data_dir=data_dir,
            from_discogs=False,
        )
        for df in (col, wl):
            if df is None or df.empty or "album_id" not in df.columns:
                continue
            for x in df["album_id"].astype(str):
                s = x.strip()
                if s.isdigit():
                    ids.add(s)
    return ids


def _grader_raw_discogs_dir(repo: Path, grader_config: Path) -> Path:
    with open(grader_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = (cfg.get("paths") or {}).get("raw", "grader/data/raw/")
    base = _resolve_path(repo, raw)
    return base / "discogs"


def _collect_grader_inventory_release_ids(inventory_root: Path) -> set[str]:
    ids: set[str] = set()
    if not inventory_root.is_dir():
        return ids
    for path in sorted(inventory_root.rglob("page_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for listing in data.get("listings") or []:
            if not isinstance(listing, dict):
                continue
            rel = listing.get("release")
            rid = None
            if isinstance(rel, dict):
                rid = rel.get("id")
            if rid is None:
                continue
            s = str(rid).strip()
            if s.isdigit():
                ids.add(s)
    return ids


def _collect_optional_jsonl_release_ids(paths: Iterable[Path]) -> set[str]:
    """Pick up ``release_id`` / ``discogs_release_id`` if present on any line."""
    keys = ("release_id", "discogs_release_id")
    ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                for k in keys:
                    v = obj.get(k)
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s.isdigit():
                        ids.add(s)
                        break
    return ids


def _sort_ids(ids: set[str]) -> list[str]:
    return sorted(ids, key=lambda x: int(x))


def main() -> int:
    repo = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Merge unique Discogs release IDs from recommender collection/wantlist "
            "and grader Discogs inventory cache."
        ),
    )
    parser.add_argument(
        "--recommender-config",
        type=Path,
        default=repo / "recommender" / "configs" / "base.yaml",
        help="YAML with discogs.usernames (default: recommender/configs/base.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Recommender CSV fallback directory (collection.csv / wantlist.csv)",
    )
    parser.add_argument(
        "--collection-csv",
        type=Path,
        default=None,
        help="Override collection CSV path",
    )
    parser.add_argument(
        "--wantlist-csv",
        type=Path,
        default=None,
        help="Override wantlist CSV path",
    )
    parser.add_argument(
        "--grader-config",
        type=Path,
        default=repo / "grader" / "configs" / "grader.yaml",
        help="Grader YAML (paths.raw → …/discogs/inventory)",
    )
    parser.add_argument(
        "--grader-inventory-root",
        type=Path,
        default=None,
        help="Override inventory root (default: <paths.raw>/discogs/inventory)",
    )
    parser.add_argument(
        "--no-grader-splits",
        action="store_true",
        help="Do not scan grader split JSONL for optional release_id fields",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo / "price_estimator" / "data" / "raw" / "recommender_grader_release_ids.txt",
        help="Output file (one release_id per line)",
    )
    args = parser.parse_args()

    rcfg = _resolve_path(repo, args.recommender_config)
    usernames, cfg_data_dir = _load_recommender_usernames_and_data_dir(repo, rcfg)
    data_dir = args.data_dir
    if data_dir is not None:
        data_dir = _resolve_path(repo, data_dir)
    elif cfg_data_dir is not None:
        data_dir = cfg_data_dir

    collection_csv = (
        _resolve_path(repo, args.collection_csv) if args.collection_csv else None
    )
    wantlist_csv = (
        _resolve_path(repo, args.wantlist_csv) if args.wantlist_csv else None
    )

    recommender_ids = _collect_recommender_ids(
        usernames=usernames,
        data_dir=data_dir,
        collection_csv=collection_csv,
        wantlist_csv=wantlist_csv,
    )

    gcfg = _resolve_path(repo, args.grader_config)
    discogs_raw = _grader_raw_discogs_dir(repo, gcfg)
    inv_root = args.grader_inventory_root
    if inv_root is not None:
        inv_root = _resolve_path(repo, inv_root)
    else:
        inv_root = discogs_raw / "inventory"

    grader_inv_ids = _collect_grader_inventory_release_ids(inv_root)

    split_ids: set[str] = set()
    if not args.no_grader_splits:
        with open(gcfg, encoding="utf-8") as f:
            gyaml = yaml.safe_load(f) or {}
        splits_dir = (gyaml.get("paths") or {}).get("splits", "grader/data/splits/")
        splits_path = _resolve_path(repo, splits_dir)
        if splits_path.is_dir():
            split_ids = _collect_optional_jsonl_release_ids(sorted(splits_path.glob("*.jsonl")))

    all_ids = recommender_ids | grader_inv_ids | split_ids
    if not all_ids:
        print(
            "No release IDs found. Set DISCOGS_USER_TOKEN or DISCOGS_TOKEN, "
            "ensure recommender config lists discogs.usernames, "
            "and/or add data/raw CSVs or grader inventory cache.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ordered = _sort_ids(all_ids)
    with open(args.out, "w", encoding="utf-8") as f:
        for rid in ordered:
            f.write(rid + "\n")

    print(
        f"Wrote {len(ordered)} unique release_id lines → {args.out}\n"
        f"  recommender (collection+wantlist): {len(recommender_ids)}\n"
        f"  grader inventory JSON pages:       {len(grader_inv_ids)}\n"
        f"  grader splits (optional fields):   {len(split_ids)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
