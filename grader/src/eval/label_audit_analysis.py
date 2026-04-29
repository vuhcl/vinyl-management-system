from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_reviewed(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT source,target,assigned_label,final_label,human_action
            FROM queue
            WHERE COALESCE(human_action, '') <> ''
            """,
            conn,
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Analyze accepted label-audit corrections "
            "and plot by source."
        )
    )
    p.add_argument("--db", default="grader/reports/label_audit_queue.sqlite")
    p.add_argument(
        "--out-csv",
        default=(
            "grader/reports/"
            "label_audit_correction_rate_by_source_grade.csv"
        ),
    )
    p.add_argument(
        "--out-plot",
        default="grader/reports/label_audit_correction_rate_by_source.png",
    )
    args = p.parse_args()

    df = load_reviewed(Path(args.db))
    if df.empty:
        print("No reviewed rows found.")
        return 0

    df["is_change"] = (
        df["human_action"].isin(["accept_llm", "accept_edit"])
        & (df["final_label"].fillna("") != "")
        & (df["final_label"].fillna("") != df["assigned_label"].fillna(""))
    )
    grp = (
        df.groupby(["source", "target", "assigned_label"], dropna=False)
        .agg(
            n_rows=("is_change", "size"),
            n_changed=("is_change", "sum"),
        )
        .reset_index()
    )
    grp["change_rate"] = grp["n_changed"] / grp["n_rows"].clip(lower=1)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(out_csv, index=False)

    src = (
        grp.groupby("source", dropna=False)
        .agg(change_rate=("n_changed", "sum"), n=("n_rows", "sum"))
        .reset_index()
    )
    src["change_rate"] = src["change_rate"] / src["n"].clip(lower=1)

    plt.figure(figsize=(7, 4))
    plt.bar(src["source"], src["change_rate"])
    plt.ylabel("Correction rate")
    plt.xlabel("Source")
    plt.title("Accepted correction rate by source")
    plt.ylim(0, 1)
    plt.tight_layout()
    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=160)
    print(
        {
            "csv": str(out_csv),
            "plot": str(out_plot),
            "rows": int(len(df)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
