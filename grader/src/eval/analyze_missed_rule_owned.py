"""
Mine rule-owned false negatives for pattern review.

A rule-owned false negative is a row where ``y_true`` is a rule-owned
grade (e.g. ``Poor`` / ``Generic`` per ``grade_owners``) but the
**final** rule-adjusted prediction (``y_pred_after``) is not that
grade. These rows point at either (a) a missing hard pattern or
(b) an over-eager forbidden that blocks a correct hit — the two
``hard_signal_pre_forbidden`` / ``hard_signal_post_forbidden`` booleans
disambiguate.

Sibling of :mod:`grader.src.eval.analyze_harmful_overrides` — same
prediction path, different slice of interest.

Examples (repo root):

    uv run python -m grader.src.eval.analyze_missed_rule_owned \\
        --split test --target sleeve

    uv run python -m grader.src.eval.analyze_missed_rule_owned \\
        --split test_thin --target media
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from grader.src.config_io import load_yaml_mapping

from grader.src.evaluation.grade_analysis import resolve_rule_owned_grades
from grader.src.models.transformer import TransformerTrainer
from grader.src.pipeline import Pipeline
from grader.src.rules.rule_engine import RuleEngine


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _missed_mask(
    y_true_labels: list[str],
    y_after_labels: list[str],
    rule_owned_grades: list[str],
) -> np.ndarray:
    """
    Rows where the gold label is rule-owned but the rule-adjusted
    prediction disagrees with it. Treat unknown gold labels as not
    rule-owned (mask is False).
    """
    owned = set(rule_owned_grades)
    return np.array(
        [
            (y_t in owned) and (y_t != y_a)
            for y_t, y_a in zip(y_true_labels, y_after_labels)
        ],
        dtype=bool,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export rule-owned false negatives (gold=Poor/Generic, "
            "rule did not fire) for guideline mining."
        )
    )
    p.add_argument("--config", default="grader/configs/grader.yaml")
    p.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=("test", "test_thin"),
    )
    p.add_argument(
        "--target",
        default="sleeve",
        choices=("sleeve", "media"),
    )
    p.add_argument(
        "--artifact-subdir",
        default="",
        help="Subpath under paths.artifacts for transformer weights",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help=(
            "Per-row CSV (default: "
            "reports/missed_rule_owned_<split>_<target>.csv)"
        ),
    )
    p.add_argument(
        "--output-patterns",
        default="",
        help=(
            "Text histogram (default: "
            "reports/missed_rule_owned_patterns_<split>_<target>.txt)"
        ),
    )
    p.add_argument(
        "--max-csv-rows",
        type=int,
        default=5000,
        help="Cap rows written to CSV (all still counted for patterns)",
    )
    args = p.parse_args()

    cfg = load_yaml_mapping(args.config)
    guidelines = load_yaml_mapping(args.guidelines)

    rule_owned = resolve_rule_owned_grades(guidelines).get(args.target, [])
    if not rule_owned:
        raise SystemExit(
            f"No rule-owned grades for target={args.target!r} — nothing "
            "to mine."
        )

    splits_dir = Path(cfg["paths"]["splits"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise SystemExit(f"Split not found: {split_path}")

    records = _load_jsonl(split_path)
    texts = [r.get("text_clean") or r.get("text") or "" for r in records]
    item_ids = [str(r.get("item_id", i)) for i, r in enumerate(records)]

    subdir = args.artifact_subdir.strip() or None
    trainer = TransformerTrainer(
        config_path=args.config, artifact_subdir=subdir
    )
    trainer.encoders = trainer.load_encoders()
    trainer.load_model()

    raw = trainer.predict(texts=texts, item_ids=item_ids, records=records)
    Pipeline._merge_description_metadata(raw, records)
    rules_cfg = cfg.get("rules") or {}
    allow_ex = bool(rules_cfg.get("allow_excellent_soft_override", False))
    engine = RuleEngine(
        guidelines_path=args.guidelines,
        allow_excellent_soft_override=allow_ex,
    )
    adjusted = engine.apply_batch(raw, texts)

    pred_key = f"predicted_{args.target}_condition"
    true_key = f"{args.target}_label"
    y_true = [str(r.get(true_key, "")) for r in records]
    y_before = [str(x[pred_key]) for x in raw]
    y_after = [str(x[pred_key]) for x in adjusted]

    mask = _missed_mask(y_true, y_after, rule_owned)

    csv_path = (
        Path(args.output_csv)
        if args.output_csv.strip()
        else reports_dir
        / f"missed_rule_owned_{args.split}_{args.target}.csv"
    )
    pat_path = (
        Path(args.output_patterns)
        if args.output_patterns.strip()
        else reports_dir
        / f"missed_rule_owned_patterns_{args.split}_{args.target}.txt"
    )

    class_counts: Counter[str] = Counter()
    triplet_counts: Counter[str] = Counter()
    pre_only_counts: Counter[str] = Counter()
    neither_counts: Counter[str] = Counter()
    csv_rows: list[dict[str, Any]] = []

    for i in np.flatnonzero(mask):
        idx = int(i)
        r = records[idx]
        text = texts[idx]
        gold = y_true[idx]
        pre_forbidden = engine.would_hard_signal_match(
            text, args.target, gold
        )
        post_forbidden = engine.would_hard_override_fire(
            text, args.target, gold
        )
        row = {
            "target": args.target,
            "item_id": str(r.get("item_id", idx)),
            "true_label": gold,
            "sleeve_label": r.get("sleeve_label", ""),
            "media_label": r.get("media_label", ""),
            "y_pred_before": y_before[idx],
            "y_pred_after": y_after[idx],
            "hard_signal_pre_forbidden": pre_forbidden,
            "hard_signal_post_forbidden": post_forbidden,
            "text": r.get("text") or "",
            "text_clean": text,
        }
        csv_rows.append(row)

        class_counts[gold] += 1
        triplet = (
            f"{gold} | model={y_before[idx]} | rule={y_after[idx]}"
        )
        triplet_counts[triplet] += 1
        if pre_forbidden and not post_forbidden:
            pre_only_counts[gold] += 1
        elif not pre_forbidden and not post_forbidden:
            neither_counts[gold] += 1

    lines = [
        f"Missed rule-owned — split={args.split} target={args.target}",
        f"Rule-owned grades: {rule_owned}",
        f"Total missed rows: {len(csv_rows)}",
        "",
        "Counts by true label:",
    ]
    for lab, c in class_counts.most_common():
        lines.append(f"  {c:5d}  {lab}")

    lines.extend(
        [
            "",
            "Blocked-by-forbidden (pre=True, post=False) — "
            "'over-eager forbidden' candidates:",
        ]
    )
    for lab, c in pre_only_counts.most_common():
        lines.append(f"  {c:5d}  {lab}")

    lines.extend(
        [
            "",
            "No hard-signal match (pre=False, post=False) — "
            "'missing pattern' candidates:",
        ]
    )
    for lab, c in neither_counts.most_common():
        lines.append(f"  {c:5d}  {lab}")

    lines.extend(
        [
            "",
            "Top transitions (true_label | model=… | rule=…):",
        ]
    )
    for tri, c in triplet_counts.most_common(80):
        lines.append(f"  {c:5d}  {tri}")

    pat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote pattern summary: {pat_path}")

    csv_rows.sort(key=lambda x: (x["true_label"], x["item_id"]))
    cap = min(len(csv_rows), args.max_csv_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "item_id",
        "true_label",
        "sleeve_label",
        "media_label",
        "y_pred_before",
        "y_pred_after",
        "hard_signal_pre_forbidden",
        "hard_signal_post_forbidden",
        "text",
        "text_clean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows[:cap]:
            w.writerow(row)
    print(
        f"Wrote {cap} / {len(csv_rows)} missed rule-owned rows to "
        f"{csv_path} (cap max-csv-rows={args.max_csv_rows})"
    )


if __name__ == "__main__":
    main()
