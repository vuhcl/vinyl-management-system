#!/usr/bin/env python3
"""
Build a zip of grader paths needed for Colab Workflow A (see notebooks/colab_vinyl_grader_train.ipynb).

Default layout matches grader.yaml when extracted at the **repo root**:
  grader/data/splits/*.jsonl
  grader/artifacts/...   (baseline, TF-IDF, features, encoders; see flags)

By default EXCLUDES (usually huge or recreated on Colab):
  - grader/artifacts/tuning/
  - grader/artifacts/transformer_weights.pt

Examples:
  python scripts/zip_grader_colab_artifacts.py
  python scripts/zip_grader_colab_artifacts.py -o ~/Desktop/grader_colab.zip
  python scripts/zip_grader_colab_artifacts.py --include-transformer-weights
  python scripts/zip_grader_colab_artifacts.py --include-tuning
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output zip path (default: <repo>/grader_colab_workflow_a.zip)",
    )
    ap.add_argument(
        "--include-transformer-weights",
        action="store_true",
        help="Include grader/artifacts/transformer_weights.pt (~250MB+).",
    )
    ap.add_argument(
        "--include-tuning",
        action="store_true",
        help="Include grader/artifacts/tuning/ (often 1GB+).",
    )
    args = ap.parse_args()

    root = _repo_root()
    splits = root / "grader" / "data" / "splits"
    artifacts = root / "grader" / "artifacts"

    if not splits.is_dir():
        print(f"ERROR: missing splits dir: {splits}", file=sys.stderr)
        return 1
    if not artifacts.is_dir():
        print(f"ERROR: missing artifacts dir: {artifacts}", file=sys.stderr)
        return 1

    out = args.output or (root / "grader_colab_workflow_a.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    skip_prefixes: list[Path] = []
    if not args.include_tuning:
        skip_prefixes.append(artifacts / "tuning")

    skip_files: set[Path] = set()
    if not args.include_transformer_weights:
        skip_files.add(artifacts / "transformer_weights.pt")

    def skip_path(p: Path) -> bool:
        rp = p.resolve()
        if rp in skip_files:
            return True
        for pref in skip_prefixes:
            try:
                rp.relative_to(pref.resolve())
                return True
            except ValueError:
                continue
        return False

    files: list[Path] = []
    for base in (splits, artifacts):
        if base.is_file():
            files.append(base)
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            if skip_path(path):
                continue
            files.append(path)

    if not files:
        print("ERROR: no files matched — check paths and filters.", file=sys.stderr)
        return 1

    written = 0
    with zipfile.ZipFile(
        out,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for path in files:
            arc = path.relative_to(root).as_posix()
            zf.write(path, arcname=arc)
            written += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({written} files, {size_mb:.1f} MiB)")
    print("Upload this file to Google Drive and set ZIP in the Colab notebook cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
