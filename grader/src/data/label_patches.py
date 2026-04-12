"""
Apply hand-maintained label corrections to ingested JSONL.

``data.label_patches_path`` in grader.yaml should point to a JSONL file (see
``grader/data/label_patches.example.jsonl``). After Discogs/eBay ingest overwrites
``discogs_processed.jsonl`` / ``ebay_processed.jsonl``, the training pipeline
re-applies those patches so manual relabels survive re-ingestion.

Each patch line must include ``item_id`` and ``source`` (``discogs`` or
``ebay_jp``). Optional keys merged into the matching row (whitelist only):
``sleeve_label``, ``media_label``, ``text``, ``label_confidence``,
``media_verifiable``, ``raw_sleeve``, ``raw_media``, ``obi_condition``.
"""
from __future__ import annotations

import argparse
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

_VALID_SOURCES: frozenset[str] = frozenset({"discogs", "ebay_jp"})


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
    processed JSONL for Discogs and eBay.

    Call this only after ingest has written (or would have written) those files.
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

    d_stats = apply_label_patches_to_processed_file(
        discogs_path, source="discogs", index=index, dry_run=dry_run
    )
    e_stats = apply_label_patches_to_processed_file(
        ebay_path, source="ebay_jp", index=index, dry_run=dry_run
    )
    total_updated = int(d_stats.get("updated", 0)) + int(e_stats.get("updated", 0))
    return {
        "enabled": True,
        "path": str(path),
        "patch_lines_loaded": len(patches),
        "patch_index_size": len(index),
        "skipped_invalid_entries": skipped,
        "discogs": d_stats,
        "ebay_jp": e_stats,
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
            if src not in _VALID_SOURCES or not iid:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply label_patches_path to processed JSONL, or export patches "
            "from unified/processed JSONL."
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

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
