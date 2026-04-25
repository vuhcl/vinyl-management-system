"""
Print seller comments for rows where ``clean_text`` drops a protected
whole-token match (same check as ``Preprocessor.process_record``).

Examples (repo root, after ``unified.jsonl`` exists under ``paths.processed``):

    uv run python -m grader.src.eval.report_protected_terms_lost_comments

    uv run python -m grader.src.eval.report_protected_terms_lost_comments \\
        --input-jsonl grader/data/processed/unified.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from grader.src.data.preprocess import Preprocessor


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected mapping YAML at {path}")
    return data


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print raw comments for listings that lose protected terms "
            "in clean_text."
        )
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Grader config YAML (default: grader/configs/grader.yaml)",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Grading guidelines YAML",
    )
    parser.add_argument(
        "--input-jsonl",
        default="",
        help=(
            "Override unified JSONL path (default: "
            "paths.processed/unified.jsonl from config)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many loss rows (0 = no limit)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    gl_path = Path(args.guidelines)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path.resolve()}")
    if not gl_path.is_file():
        raise SystemExit(f"Guidelines not found: {gl_path.resolve()}")

    config = _load_yaml(cfg_path)
    if args.input_jsonl:
        in_path = Path(args.input_jsonl)
    else:
        processed = Path(config["paths"]["processed"])
        in_path = processed / "unified.jsonl"

    if not in_path.is_file():
        raise SystemExit(
            f"Input JSONL not found: {in_path.resolve()}\n"
            "Pass --input-jsonl or run harmonize so unified.jsonl exists."
        )

    pre = Preprocessor(
        config_path=str(cfg_path),
        guidelines_path=str(gl_path),
        config=config,
    )

    n_seen = 0
    n_lost = 0
    for rec in _iter_jsonl(in_path):
        n_seen += 1
        raw = str(rec.get("text") or "")
        cleaned = pre.clean_text(raw)
        lost = pre._verify_protected_terms(raw, cleaned)
        if not lost:
            continue
        n_lost += 1
        item_id = rec.get("item_id", "?")
        print("=" * 80, file=sys.stdout)
        print(f"item_id: {item_id}", file=sys.stdout)
        print(f"lost_terms: {lost}", file=sys.stdout)
        print("--- raw text ---", file=sys.stdout)
        print(raw, file=sys.stdout)
        print("--- text_clean ---", file=sys.stdout)
        print(cleaned, file=sys.stdout)
        if args.limit and n_lost >= args.limit:
            break

    print("=" * 80, file=sys.stdout)
    print(
        f"Done. rows_scanned={n_seen} rows_with_protected_loss={n_lost}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
