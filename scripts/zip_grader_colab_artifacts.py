#!/usr/bin/env python3
"""
Zip grader splits + on-disk TF-IDF / baseline artifacts for Colab training.

After unzip at repo root (see ``notebooks/colab_vinyl_grader_train.ipynb``):

  grader/data/splits/*.jsonl
  grader/artifacts/features/**
  grader/artifacts/label_encoder_*.pkl
  grader/artifacts/tfidf_vectorizer_*.pkl, preprocessor.pkl
  grader/artifacts/baseline_*.pkl

Run from repo root after local ``pipeline train`` has built splits and
``tfidf_features`` (or baseline-only train).

  uv run python scripts/zip_grader_colab_artifacts.py -o grader_colab_artifacts.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

# Root-level pickles to pack (omit tuning/ unless --include-tuning).
_ARTIFACT_ROOT_PICKLE_PREFIXES = (
    "label_encoder_",
    "tfidf_vectorizer_",
    "baseline_",
)
_ARTIFACT_ROOT_PICKLE_NAMES = frozenset({"preprocessor.pkl"})


def _load_paths(config_path: Path) -> tuple[Path, Path]:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths") or {}
    splits = paths.get("splits", "grader/data/splits/")
    artifacts = paths.get("artifacts", "grader/artifacts/")
    return Path(str(splits)), Path(str(artifacts))


def _collect_files(
    repo_root: Path, splits_dir: Path, artifacts_dir: Path
) -> list[Path]:
    files: list[Path] = []
    sd = (repo_root / splits_dir).resolve()
    if not sd.is_dir():
        raise FileNotFoundError(f"Splits directory missing: {sd}")
    for p in sorted(sd.glob("*.jsonl")):
        files.append(p)

    ad = (repo_root / artifacts_dir).resolve()
    feat = ad / "features"
    if not feat.is_dir():
        raise FileNotFoundError(
            f"Features directory missing: {feat} — "
            "run TF-IDF feature build locally first."
        )
    for p in sorted(feat.rglob("*")):
        if p.is_file():
            files.append(p)

    if not ad.is_dir():
        raise FileNotFoundError(f"Artifacts directory missing: {ad}")

    for p in sorted(ad.glob("*.pkl")):
        if p.name in _ARTIFACT_ROOT_PICKLE_NAMES:
            files.append(p)
            continue
        if any(
            p.name.startswith(prefix) for prefix in _ARTIFACT_ROOT_PICKLE_PREFIXES
        ):
            files.append(p)

    return files


def _collect_tuning(repo_root: Path, artifacts_dir: Path) -> list[Path]:
    tuning = (repo_root / artifacts_dir / "tuning").resolve()
    if not tuning.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(tuning.rglob("*")):
        if p.is_file():
            out.append(p)
    return out


def _validate_required(
    repo_root: Path, splits_dir: Path, artifacts_dir: Path
) -> None:
    sd = (repo_root / splits_dir).resolve()
    ad = (repo_root / artifacts_dir).resolve()
    missing: list[str] = []
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        if not (sd / name).is_file():
            missing.append(str(sd / name))
    for enc in ("label_encoder_sleeve.pkl", "label_encoder_media.pkl"):
        if not (ad / enc).is_file():
            missing.append(str(ad / enc))
    for tv in ("tfidf_vectorizer_sleeve.pkl", "tfidf_vectorizer_media.pkl"):
        if not (ad / tv).is_file():
            missing.append(str(ad / tv))
    for target in ("sleeve", "media"):
        for suffix in ("", "_calibrated"):
            name = f"baseline_{target}{suffix}.pkl"
            if not (ad / name).is_file():
                missing.append(str(ad / name))
    if missing:
        raise FileNotFoundError(
            "Missing required Colab inputs:\n  "
            + "\n  ".join(missing)
            + "\n\nTrain baseline + TF-IDF features locally, "
            "then re-run this script."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current working directory)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("grader/configs/grader.yaml"),
        help="Grader config (for paths.splits and paths.artifacts)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("grader_colab_artifacts.zip"),
        help="Output zip path (default: grader_colab_artifacts.zip under cwd)",
    )
    parser.add_argument(
        "--include-tuning",
        action="store_true",
        help="Also pack grader/artifacts/tuning/** if present (large).",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help=(
            "Do not require baseline + encoder files "
            "(for partial / tune-only zips)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be zipped, do not write zip.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.config.is_absolute():
        config_path = args.config
    else:
        config_path = (repo_root / args.config).resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    splits_dir, artifacts_dir = _load_paths(config_path)
    try:
        if not args.skip_validate:
            _validate_required(repo_root, splits_dir, artifacts_dir)
        paths = _collect_files(repo_root, splits_dir, artifacts_dir)
        if args.include_tuning:
            paths.extend(_collect_tuning(repo_root, artifacts_dir))
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Stable order, de-dupe (e.g. if tuning overlaps — it should not)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            ordered.append(rp)

    out_zip = args.output
    if not out_zip.is_absolute():
        out_zip = (repo_root / out_zip).resolve()

    print(f"Repo root:     {repo_root}")
    print(f"Config:        {config_path}")
    print(f"Files to pack: {len(ordered)}")
    if args.dry_run:
        for p in ordered[:50]:
            print(" ", p.relative_to(repo_root))
        if len(ordered) > 50:
            print(f"  ... and {len(ordered) - 50} more")
        return 0

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for p in ordered:
            arc = p.relative_to(repo_root)
            zf.write(p, arcname=str(arc).replace("\\", "/"))
    mib = out_zip.stat().st_size // 1024 // 1024
    print(f"Wrote {out_zip} ({mib} MiB approx)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
