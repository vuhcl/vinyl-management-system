from __future__ import annotations

import argparse
from pathlib import Path

from grader.src.eval.label_audit_backend import commit_queue_to_label_patches


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Commit reviewed label-audit decisions into "
            "label_patches JSONL."
        )
    )
    p.add_argument("--db", default="grader/reports/label_audit_queue.sqlite")
    p.add_argument(
        "--label-patches",
        default="grader/data/label_patches.jsonl",
        help="Destination label_patches JSONL.",
    )
    p.add_argument(
        "--temp-csv",
        default="grader/reports/label_audit_commit_preview.csv",
        help="Temporary grouped CSV used for append_csv_to_label_patches.",
    )
    args = p.parse_args()
    out = commit_queue_to_label_patches(
        db_path=Path(args.db),
        label_patches_path=Path(args.label_patches),
        temp_csv_path=Path(args.temp_csv),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
