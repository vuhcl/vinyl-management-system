from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from grader.src.eval.label_audit_backend import (
    build_queue_from_cleanlab_csvs,
    ensure_db,
)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build label-audit SQLite queue from cleanlab CSV files."
    )
    p.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML.",
    )
    p.add_argument(
        "--db",
        default="grader/reports/label_audit_queue.sqlite",
        help="Output SQLite DB path.",
    )
    p.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more cleanlab audit CSV files.",
    )
    p.add_argument(
        "--default-split",
        default="train",
        choices=["train", "val", "test"],
        help="Fallback split when not inferable from CSV filename/column.",
    )
    args = p.parse_args()

    cfg = _load_yaml(Path(args.config))
    splits_dir = Path(cfg["paths"]["splits"])
    db_path = Path(args.db)
    ensure_db(db_path)
    stats = build_queue_from_cleanlab_csvs(
        db_path=db_path,
        csv_paths=[Path(x) for x in args.csv],
        splits_dir=splits_dir,
        default_split=args.default_split,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
