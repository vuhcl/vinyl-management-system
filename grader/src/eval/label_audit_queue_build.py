from __future__ import annotations

import argparse
from pathlib import Path

from grader.src.eval.cli_common import (
    add_grader_config_arg,
    load_grader_config_mapping,
)
from grader.src.eval.label_audit_backend import (
    build_queue_from_cleanlab_csvs,
    ensure_db,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build label-audit SQLite queue from cleanlab CSV files."
    )
    add_grader_config_arg(
        p,
        help_text="Path to grader config YAML.",
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

    cfg = load_grader_config_mapping(args.config)
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
