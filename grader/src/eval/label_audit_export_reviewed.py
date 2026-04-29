from __future__ import annotations

import argparse
from pathlib import Path

from grader.src.eval.label_audit_backend import export_reviewed_to_csv


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export reviewed rows from queue DB to CSV."
    )
    p.add_argument("--db", default="grader/reports/label_audit_queue.sqlite")
    p.add_argument(
        "--output", default="grader/reports/label_audit_reviewed_export.csv"
    )
    args = p.parse_args()
    n = export_reviewed_to_csv(Path(args.db), Path(args.output))
    print({"rows": n, "output": args.output})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
