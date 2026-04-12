"""
Mine harmful rule overrides (correct model → wrong after rules) for pattern review.

Examples (repo root):

    uv run python -m grader.src.eval.analyze_harmful_overrides \\
        --split test --output-csv grader/reports/harmful_overrides_test.csv

    uv run python -m grader.src.eval.analyze_harmful_overrides --split test_thin
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from grader.src.evaluation.metrics import remap_true_and_encode_predictions
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.transformer import TransformerTrainer
from grader.src.pipeline import Pipeline
from grader.src.rules.rule_engine import RuleEngine


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(__import__("json").loads(line))
    return rows


def _harmful_mask(
    y_true: np.ndarray,
    class_names: np.ndarray,
    before: list[str],
    after: list[str],
) -> np.ndarray:
    """Rows where label changed, was correct before rules, wrong after."""
    y_t2, _, (y_b, y_a) = remap_true_and_encode_predictions(
        y_true, class_names, before, after
    )
    correct_b = y_b == y_t2
    correct_a = y_a == y_t2
    changed = np.array(
        [str(a) != str(b) for a, b in zip(before, after)],
        dtype=bool,
    )
    return changed & correct_b & ~correct_a


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export harmful rule overrides and transition histograms"
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
        "--artifact-subdir",
        default="",
        help="Subpath under paths.artifacts for transformer weights",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help="Per-row harmful export (default: reports/harmful_overrides_<split>.csv)",
    )
    p.add_argument(
        "--output-patterns",
        default="",
        help="Text histogram (default: reports/harmful_override_patterns_<split>.txt)",
    )
    p.add_argument(
        "--max-csv-rows",
        type=int,
        default=5000,
        help="Cap rows written to CSV (all still counted for patterns)",
    )
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
    trainer = TransformerTrainer(config_path=args.config, artifact_subdir=subdir)
    trainer.encoders = trainer.load_encoders()
    trainer.load_model()

    raw = trainer.predict(texts=texts, item_ids=item_ids, records=records)
    Pipeline._merge_description_metadata(raw, records)
    engine = RuleEngine(guidelines_path=args.guidelines)
    adjusted = engine.apply_batch(raw, texts)

    features_dir = str(Path(cfg["paths"]["artifacts"]) / "features")
    csv_path = (
        Path(args.output_csv)
        if args.output_csv.strip()
        else reports_dir / f"harmful_overrides_{args.split}.csv"
    )
    pat_path = (
        Path(args.output_patterns)
        if args.output_patterns.strip()
        else reports_dir / f"harmful_override_patterns_{args.split}.txt"
    )

    triplet_counts: Counter[str] = Counter()
    wrong_after_counts: Counter[str] = Counter()
    csv_rows: list[dict[str, Any]] = []

    for target in ("sleeve", "media"):
        enc = trainer.encoders[target]
        true_key = "sleeve_label" if target == "sleeve" else "media_label"
        # Derive true labels from the JSONL records (not the feature cache) so
        # that any manual relabels applied after the last feature build are
        # reflected correctly here.
        known = set(enc.classes_)
        y = np.array(
            [
                enc.transform([r.get(true_key)])[0]
                if r.get(true_key) in known
                else -1
                for r in records
            ]
        )
        pred_key = f"predicted_{target}_condition"
        before = [str(x[pred_key]) for x in raw]
        after = [str(x[pred_key]) for x in adjusted]
        meta_key = "metadata"
        harm = _harmful_mask(y, enc.classes_, before, after)

        for i in np.flatnonzero(harm):
            r = records[int(i)]
            triplet = (
                f"{r.get(true_key)} | model={before[i]} | rule={after[i]}"
            )
            triplet_counts[triplet] += 1
            wrong_after_counts[after[i]] += 1
            text_raw = r.get("text") or ""
            text_clean = r.get("text_clean") or r.get("text") or ""
            row = {
                "target": target,
                "item_id": str(r.get("item_id", i)),
                "true_label": r.get(true_key, ""),
                "sleeve_label": r.get("sleeve_label", ""),
                "media_label": r.get("media_label", ""),
                "model_pred": before[i],
                "rule_pred": after[i],
                "model_sleeve": str(raw[i]["predicted_sleeve_condition"]),
                "model_media": str(raw[i]["predicted_media_condition"]),
                "rule_sleeve": str(adjusted[i]["predicted_sleeve_condition"]),
                "rule_media": str(adjusted[i]["predicted_media_condition"]),
                "text": text_raw,
                "text_clean": text_clean,
                "rule_override_applied": adjusted[i][meta_key].get(
                    "rule_override_applied", ""
                ),
                "override_target": adjusted[i][meta_key].get(
                    "rule_override_target", ""
                ),
                "contradiction": adjusted[i][meta_key].get(
                    "contradiction_detected", ""
                ),
            }
            csv_rows.append(row)

    lines = [
        f"Harmful overrides — split={args.split}",
        f"Total harmful rows (sleeve + media, a row may appear twice): {len(csv_rows)}",
        "",
        "Top transitions (true_label | model=… | rule=…):",
    ]
    for tri, c in triplet_counts.most_common(80):
        lines.append(f"  {c:5d}  {tri}")

    lines.extend(
        [
            "",
            "Counts by rule_pred (wrong final grade when harmful):",
        ]
    )
    for lab, c in wrong_after_counts.most_common():
        lines.append(f"  {c:5d}  {lab}")

    pat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote pattern summary: {pat_path}")

    csv_rows.sort(key=lambda x: (x["target"], x["item_id"]))
    cap = min(len(csv_rows), args.max_csv_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "item_id",
        "true_label",
        "sleeve_label",
        "media_label",
        "model_pred",
        "rule_pred",
        "model_sleeve",
        "model_media",
        "rule_sleeve",
        "rule_media",
        "rule_override_applied",
        "override_target",
        "contradiction",
        "text",
        "text_clean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in csv_rows[:cap]:
            w.writerow(row)
    print(
        f"Wrote {cap} / {len(csv_rows)} harmful rows to {csv_path} "
        f"(cap max-csv-rows={args.max_csv_rows})"
    )


if __name__ == "__main__":
    main()
