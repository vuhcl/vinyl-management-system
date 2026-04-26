"""
Apply hand-maintained label corrections to ingested JSONL.

``data.label_patches_path`` in grader.yaml should point to a JSONL file (see
``grader/data/label_patches.example.jsonl``). After Discogs/eBay ingest overwrites
``discogs_processed.jsonl`` / ``ebay_processed.jsonl`` /
``discogs_sale_history.jsonl`` (when used), the training pipeline
re-applies those patches so manual relabels survive re-ingestion.

Each patch line must include ``item_id`` and ``source`` (``discogs``,
``ebay_jp``, or ``discogs_sale_history``). Optional keys merged into the
matching row (whitelist only):
``sleeve_label``, ``media_label``, ``text``, ``label_confidence``,
``media_verifiable``, ``raw_sleeve``, ``raw_media``, ``obi_condition``.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ALLOWED_MERGE_KEYS: frozenset[str] = frozenset(
    {
        "sleeve_label",
        "media_label",
        "text",
        "label_confidence",
        "media_verifiable",
        "raw_sleeve",
        "raw_media",
        "obi_condition",
    }
)

_VALID_SOURCES: frozenset[str] = frozenset(
    {"discogs", "ebay_jp", "discogs_sale_history"}
)


def _patch_dict_from_csv_row(row: dict[str, str | None]) -> dict[str, Any] | None:
    """
    Build one patch object from a CSV row, or return None if the row is invalid
    (bad source/ids, or no allowed merge fields).
    """
    iid = str(row.get("item_id", "")).strip()
    src = str(row.get("source", "")).strip()
    if not iid or src not in _VALID_SOURCES:
        return None
    obj: dict[str, Any] = {"item_id": iid, "source": src}
    for k in _ALLOWED_MERGE_KEYS:
        if k not in row:
            continue
        raw = row.get(k)
        if raw is None:
            continue
        if isinstance(raw, str) and not raw.strip():
            continue
        if k == "label_confidence":
            try:
                obj[k] = float(raw)
            except (TypeError, ValueError):
                pass
        else:
            obj[k] = raw
    if len(obj) <= 2:
        return None
    return obj


def _sleeve_label_from_csv_row(row: dict[str, str | None]) -> str | None:
    """Non-empty ``sleeve_label`` from a CSV row, or None if absent/blank."""
    raw = row.get("sleeve_label")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _resolve_path(raw: str | Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def load_label_patches(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "label_patches: skip line %d (invalid JSON): %s",
                    lineno,
                    e,
                )
                continue
            if not isinstance(obj, dict):
                logger.warning("label_patches: skip line %d (not an object)", lineno)
                continue
            rows.append(obj)
    return rows


def _build_patch_index(
    patches: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    """
    Map (source, item_id) -> merge fields. Later lines override earlier ones.
    Returns (index, n_skipped_invalid).
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    skipped = 0
    for row in patches:
        src = str(row.get("source", "")).strip()
        iid = str(row.get("item_id", "")).strip()
        if src not in _VALID_SOURCES or not iid:
            skipped += 1
            logger.warning(
                "label_patches: skip entry missing valid source/item_id: %r",
                row,
            )
            continue
        payload = {
            k: v
            for k, v in row.items()
            if k in _ALLOWED_MERGE_KEYS and v is not None
        }
        if not payload:
            skipped += 1
            logger.warning(
                "label_patches: skip entry with no allowed fields: %r",
                row,
            )
            continue
        index[(src, iid)] = payload
    return index, skipped


def _apply_index_to_records(
    records: list[dict[str, Any]],
    source: str,
    index: dict[tuple[str, str], dict[str, Any]],
) -> int:
    """Return number of rows updated."""
    updated = 0
    for rec in records:
        sid = str(rec.get("source", "")).strip()
        iid = str(rec.get("item_id", "")).strip()
        if sid != source or not iid:
            continue
        patch = index.get((source, iid))
        if not patch:
            continue
        for k, v in patch.items():
            rec[k] = v
        updated += 1
    return updated


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def apply_label_patches_to_processed_file(
    jsonl_path: Path,
    *,
    source: str,
    index: dict[tuple[str, str], dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply patches whose ``source`` matches this file. Returns small stats dict."""
    if not jsonl_path.is_file():
        return {
            "path": str(jsonl_path),
            "exists": False,
            "updated": 0,
            "orphan_keys": 0,
        }
    records = _load_jsonl(jsonl_path)
    keys_for_file = {k for k in index if k[0] == source}
    updated = _apply_index_to_records(records, source, index)
    touched_ids = {k[1] for k in keys_for_file}
    seen_ids = {str(r.get("item_id", "")) for r in records if r.get("source") == source}
    orphan = len(touched_ids - seen_ids)
    if orphan:
        logger.warning(
            "label_patches: %d patch entries for source=%r had no matching row "
            "in %s",
            orphan,
            source,
            jsonl_path,
        )
    if not dry_run and updated:
        _write_jsonl(jsonl_path, records)
        logger.info(
            "label_patches: updated %d rows in %s",
            updated,
            jsonl_path,
        )
    elif dry_run:
        logger.info(
            "label_patches (dry-run): would update %d rows in %s",
            updated,
            jsonl_path,
        )
    return {
        "path": str(jsonl_path),
        "exists": True,
        "updated": updated,
        "orphan_keys": orphan,
    }


def apply_label_patches_after_ingest(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    If ``data.label_patches_path`` is set and the file exists, merge patches into
    processed JSONL for Discogs, eBay JP, and (when present) Discogs sale history.

    Call only after the writers for those JSONLs have run in the current session
    (training pipeline runs sale-history export before Discogs/eBay ingest so
    ``discogs_sale_history.jsonl`` is not overwritten after patching).
    """
    data_cfg = config.get("data") or {}
    raw = data_cfg.get("label_patches_path")
    if raw is None or str(raw).strip() == "":
        return {
            "enabled": False,
            "hint": "Set data.label_patches_path in grader.yaml to a JSONL file "
            "(see grader/data/label_patches.example.jsonl).",
        }

    path = _resolve_path(str(raw).strip())
    if not path.is_file():
        logger.warning("label_patches_path does not exist — skipping: %s", path)
        return {"enabled": True, "path": str(path), "missing_file": True}

    patches = load_label_patches(path)
    index, skipped = _build_patch_index(patches)
    processed_dir = _resolve_path(config["paths"]["processed"])
    discogs_path = processed_dir / "discogs_processed.jsonl"
    ebay_path = processed_dir / "ebay_processed.jsonl"
    sale_path = processed_dir / "discogs_sale_history.jsonl"

    d_stats = apply_label_patches_to_processed_file(
        discogs_path, source="discogs", index=index, dry_run=dry_run
    )
    e_stats = apply_label_patches_to_processed_file(
        ebay_path, source="ebay_jp", index=index, dry_run=dry_run
    )
    sh_stats = apply_label_patches_to_processed_file(
        sale_path, source="discogs_sale_history", index=index, dry_run=dry_run
    )
    total_updated = (
        int(d_stats.get("updated", 0))
        + int(e_stats.get("updated", 0))
        + int(sh_stats.get("updated", 0))
    )
    return {
        "enabled": True,
        "path": str(path),
        "patch_lines_loaded": len(patches),
        "patch_index_size": len(index),
        "skipped_invalid_entries": skipped,
        "discogs": d_stats,
        "ebay_jp": e_stats,
        "discogs_sale_history": sh_stats,
        "updated_total": total_updated,
        "dry_run": dry_run,
    }


def export_label_patches_from_jsonl(
    input_path: Path,
    output_path: Path,
) -> dict[str, int | str]:
    """
    Build a label patch JSONL from unified or processed JSONL (one row per record).

    Each output line: ``item_id``, ``source``, ``sleeve_label``, ``media_label``.
    Rows missing any of those fields are skipped.
    """
    n_out = 0
    n_skip = 0
    out_lines: list[str] = []
    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("export: skip line %d (invalid JSON)", lineno)
                n_skip += 1
                continue
            if not isinstance(rec, dict):
                n_skip += 1
                continue
            iid = str(rec.get("item_id", "")).strip()
            src = str(rec.get("source", "")).strip()
            if str(src).strip() not in _VALID_SOURCES or not iid:
                n_skip += 1
                continue
            sl = rec.get("sleeve_label")
            ml = rec.get("media_label")
            if sl is None or ml is None:
                n_skip += 1
                continue
            if isinstance(sl, str) and not sl.strip():
                n_skip += 1
                continue
            if isinstance(ml, str) and not ml.strip():
                n_skip += 1
                continue
            patch = {
                "item_id": iid,
                "source": src,
                "sleeve_label": sl,
                "media_label": ml,
            }
            out_lines.append(json.dumps(patch, ensure_ascii=False))
            n_out += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(out_lines) + ("\n" if out_lines else ""),
        encoding="utf-8",
    )
    logger.info(
        "Exported %d label patch line(s) %s -> %s (skipped %d)",
        n_out,
        input_path,
        output_path,
        n_skip,
    )
    return {
        "exported": n_out,
        "skipped": n_skip,
        "input": str(input_path),
        "output": str(output_path),
    }


def append_csv_to_label_patches(
    csv_path: Path,
    dest_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """
    Read a CSV with at least ``item_id`` and ``source`` columns plus any
    ``_ALLOWED_MERGE_KEYS`` columns, and append one JSON patch object per row
    to ``dest_path`` (UTF-8, newline-delimited).

    ``source`` must be ``discogs``, ``ebay_jp``, or ``discogs_sale_history``.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    n_skip = 0
    out_lines: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {str(c).strip() for c in (reader.fieldnames or [])}
        if not {"item_id", "source"}.issubset(fields):
            raise ValueError(
                f"CSV must include item_id and source columns; got {reader.fieldnames!r}"
            )
        for row in reader:
            obj = _patch_dict_from_csv_row(row)
            if obj is None:
                n_skip += 1
                continue
            out_lines.append(json.dumps(obj, ensure_ascii=False))
            n_out += 1
    if not dry_run and out_lines:
        with dest_path.open("a", encoding="utf-8") as out:
            for line in out_lines:
                out.write(line + "\n")
    logger.info(
        "label_patches CSV append: %d line(s) -> %s (skipped %d, dry_run=%s)",
        n_out,
        dest_path,
        n_skip,
        dry_run,
    )
    return {
        "appended": n_out,
        "skipped_rows": n_skip,
        "csv": str(csv_path),
        "dest": str(dest_path),
        "dry_run": dry_run,
    }


def merge_csv_into_label_patches(
    csv_path: Path,
    dest_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """
    Read a CSV (``item_id``, ``source``, ``sleeve_label``, other merge columns).

    For each row, if ``(source, item_id)`` already appears in ``dest_path``,
    set ``sleeve_label`` on those JSONL line(s) from the CSV and leave all other
    fields unchanged. Otherwise, append a full patch line (same rules as
    :func:`append_csv_to_label_patches`). Rows with no new ``sleeve_label`` to
    apply to an existing key are skipped for the update path.

    The destination file is rewritten in order; comments are not preserved.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = (
        load_label_patches(dest_path) if dest_path.is_file() else []
    )
    key_to_indices: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(records):
        src = str(r.get("source", "")).strip()
        iid = str(r.get("item_id", "")).strip()
        if src in _VALID_SOURCES and iid:
            key_to_indices.setdefault((src, iid), []).append(i)

    n_new = 0
    n_updated = 0
    n_skip = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {str(c).strip() for c in (reader.fieldnames or [])}
        if not {"item_id", "source"}.issubset(fields):
            raise ValueError(
                f"CSV must include item_id and source columns; got {reader.fieldnames!r}"
            )
        for row in reader:
            iid = str(row.get("item_id", "")).strip()
            src = str(row.get("source", "")).strip()
            if not iid or src not in _VALID_SOURCES:
                n_skip += 1
                continue
            key: tuple[str, str] = (src, iid)
            if key in key_to_indices:
                new_sl = _sleeve_label_from_csv_row(row)
                if new_sl is None:
                    n_skip += 1
                    continue
                for idx in key_to_indices[key]:
                    records[idx]["sleeve_label"] = new_sl
                n_updated += 1
            else:
                obj = _patch_dict_from_csv_row(row)
                if obj is None:
                    n_skip += 1
                    continue
                records.append(obj)
                new_idx = len(records) - 1
                key_to_indices[key] = [new_idx]
                n_new += 1

    if not dry_run and (n_new or n_updated):
        _write_jsonl(dest_path, records)

    logger.info(
        "label_patches CSV merge: new=%d updated_sleeve=%d %s (skipped %d, dry_run=%s)",
        n_new,
        n_updated,
        dest_path,
        n_skip,
        dry_run,
    )
    return {
        "appended": n_new,
        "updated_sleeve": n_updated,
        "skipped_rows": n_skip,
        "csv": str(csv_path),
        "dest": str(dest_path),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply label_patches_path to processed JSONL, export patches from "
            "unified/processed JSONL, or append patch lines from a CSV."
        ),
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Grader config YAML",
    )
    parser.add_argument(
        "--export-from",
        metavar="JSONL",
        help=(
            "Read this JSONL (e.g. grader/data/processed/unified.jsonl) and "
            "write label patch lines to --export-to or data.label_patches_path."
        ),
    )
    parser.add_argument(
        "--export-to",
        metavar="JSONL",
        default="",
        help="Output path for --export-from (required if label_patches_path unset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log counts only; do not write files",
    )
    parser.add_argument(
        "--append-from-csv",
        metavar="CSV",
        default="",
        help=(
            "Read CSV (item_id, source, sleeve_label, media_label, …) and "
            "append JSONL patch lines to --append-to."
        ),
    )
    parser.add_argument(
        "--append-to",
        metavar="JSONL",
        default="",
        help="Destination JSONL for --append-from-csv (required with that flag).",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help=(
            "With --append-from-csv: match (source, item_id); set sleeve_label in "
            "the JSONL for existing keys and append new keys only. Rewrites the file."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.append_from_csv and str(args.append_from_csv).strip():
        csv_p = _resolve_path(str(args.append_from_csv).strip())
        if not args.append_to or not str(args.append_to).strip():
            print(
                "--append-to is required with --append-from-csv.",
                file=sys.stderr,
            )
            return 1
        dest_p = _resolve_path(str(args.append_to).strip())
        try:
            if args.update_existing:
                stats = merge_csv_into_label_patches(
                    csv_p, dest_p, dry_run=args.dry_run
                )
            else:
                stats = append_csv_to_label_patches(
                    csv_p, dest_p, dry_run=args.dry_run
                )
        except (OSError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 1
        print(json.dumps(stats, indent=2))
        return 0

    if args.export_from:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        inp = _resolve_path(args.export_from.strip())
        if not inp.is_file():
            print(f"--export-from not found: {inp}", file=sys.stderr)
            return 1
        if args.export_to and str(args.export_to).strip():
            outp = _resolve_path(str(args.export_to).strip())
        else:
            raw = (cfg.get("data") or {}).get("label_patches_path")
            if raw is None or not str(raw).strip():
                print(
                    "Export needs --export-to or data.label_patches_path in config.",
                    file=sys.stderr,
                )
                return 1
            outp = _resolve_path(str(raw).strip())
        exp = export_label_patches_from_jsonl(inp, outp)
        print(json.dumps(exp, indent=2))
        return 0

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    out = apply_label_patches_after_ingest(cfg, dry_run=args.dry_run)
    print(json.dumps(out, indent=2))
    if not out.get("enabled") and out.get("hint"):
        print(out["hint"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
