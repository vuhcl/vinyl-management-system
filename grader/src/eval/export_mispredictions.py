"""
Export a random sample of misclassified rows for human pattern review.

Uses the transformer under grader/artifacts/ (or a tuning subfolder).

Example (repo root):

    .venv/bin/python -m grader.src.eval.export_mispredictions \\
        --config grader/configs/grader.yaml \\
        --split test \\
        --n 100 \\
        --output grader/reports/mispredictions_sample_test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from grader.src.models.transformer import TransformerTrainer


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _top1_and_gap(proba_row: np.ndarray) -> tuple[float, float]:
    s = np.sort(proba_row.astype(float))[::-1]
    top1 = float(s[0])
    gap = float(s[0] - s[1]) if len(s) > 1 else float("nan")
    return top1, gap


def _error_bucket(s_wrong: bool, m_wrong: bool) -> str:
    if s_wrong and m_wrong:
        return "both_wrong"
    if s_wrong:
        return "sleeve_only"
    return "media_only"


def _stratified_sample(
    errors: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Roughly balance both_wrong / sleeve_only / media_only when possible."""
    rng = random.Random(seed)
    by_b: dict[str, list] = defaultdict(list)
    for e in errors:
        by_b[e["error_bucket"]].append(e)
    keys = [k for k in ("both_wrong", "sleeve_only", "media_only") if by_b[k]]
    if not keys:
        return []
    base = n // len(keys)
    out: list[dict[str, Any]] = []
    remainder = n
    for k in keys:
        pool = by_b[k][:]
        rng.shuffle(pool)
        take = min(base, len(pool), remainder)
        out.extend(pool[:take])
        remainder -= take
    # Top up from merged pool if we have budget left
    if remainder > 0:
        used_ids = {id(x) for x in out}
        rest = [e for e in errors if id(e) not in used_ids]
        rng.shuffle(rest)
        out.extend(rest[:remainder])
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample transformer misclassifications to CSV for review"
    )
    p.add_argument("--config", default="grader/configs/grader.yaml")
    p.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test", "test_thin"),
        help="Which split JSONL to score (test_thin = inadequate-note eval set)",
    )
    p.add_argument(
        "--artifact-subdir",
        default="",
        help=(
            "Path *under* paths.artifacts (e.g. tuning/partial1_low_lr). "
            "Do not pass grader/artifacts — that is already the base."
        ),
    )
    p.add_argument("--n", type=int, default=100, help="Max rows in sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stratify",
        action="store_true",
        help="Balance error types (both vs sleeve-only vs media-only)",
    )
    p.add_argument(
        "--output",
        default="",
        help="CSV path (default: grader/reports/mispredictions_<split>.csv)",
    )
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits_dir = Path(cfg["paths"]["splits"])
    split_path = splits_dir / f"{args.split}.jsonl"
    records = _load_jsonl(split_path)

    subdir = args.artifact_subdir.strip() or None
    trainer = TransformerTrainer(
        config_path=args.config,
        artifact_subdir=subdir,
    )
    trainer.encoders = trainer.load_encoders()
    trainer.load_model()

    texts = [
        r.get("text_clean") or r.get("text") or "" for r in records
    ]
    item_ids = [str(r.get("item_id", i)) for i, r in enumerate(records)]
    preds = trainer.predict(
        texts=texts,
        item_ids=item_ids,
        records=records,
    )
    id_to_pred = {x["item_id"]: x for x in preds}

    errors: list[dict[str, Any]] = []
    for r in records:
        iid = str(r["item_id"])
        pr = id_to_pred[iid]
        ts = r["sleeve_label"]
        tm = r["media_label"]
        ps = pr["predicted_sleeve_condition"]
        pm = pr["predicted_media_condition"]
        s_wrong = ts != ps
        m_wrong = tm != pm
        if not s_wrong and not m_wrong:
            continue

        cs = pr["confidence_scores"]["sleeve"]
        cm = pr["confidence_scores"]["media"]
        # Rebuild proba vectors in encoder order for gap
        enc_s = trainer.encoders["sleeve"]
        enc_m = trainer.encoders["media"]
        s_row = np.array([cs[c] for c in enc_s.classes_])
        m_row = np.array([cm[c] for c in enc_m.classes_])
        s1, sg = _top1_and_gap(s_row)
        m1, mg = _top1_and_gap(m_row)

        ev = str(r.get("media_evidence_strength", "none"))
        meta = pr.get("metadata") or {}
        ev_scores = meta.get("media_evidence_scores", "")

        errors.append(
            {
                "item_id": iid,
                "source": r.get("source", ""),
                "error_bucket": _error_bucket(s_wrong, m_wrong),
                "true_sleeve": ts,
                "pred_sleeve": ps,
                "true_media": tm,
                "pred_media": pm,
                "sleeve_wrong": s_wrong,
                "media_wrong": m_wrong,
                "sleeve_top1_prob": round(s1, 4),
                "sleeve_top2_gap": round(sg, 4),
                "media_top1_prob": round(m1, 4),
                "media_top2_gap": round(mg, 4),
                "prob_true_sleeve": round(float(cs.get(ts, 0.0)), 4),
                "prob_true_media": round(float(cm.get(tm, 0.0)), 4),
                "media_evidence_strength": ev,
                "model_media_evidence_scores": json.dumps(ev_scores)
                if isinstance(ev_scores, dict)
                else ev_scores,
                "media_verifiable": r.get("media_verifiable", ""),
                "raw_sleeve": r.get("raw_sleeve", ""),
                "raw_media": r.get("raw_media", ""),
                "text": r.get("text", ""),
                "text_clean": r.get("text_clean", ""),
            }
        )

    rng = random.Random(args.seed)
    if args.stratify:
        cap = min(args.n, len(errors))
        sample = _stratified_sample(errors, cap, args.seed)
    else:
        if len(errors) >= args.n:
            sample = rng.sample(errors, args.n)
        else:
            sample = list(errors)

    out_path = (
        Path(args.output)
        if args.output.strip()
        else Path(cfg["paths"]["reports"])
        / f"mispredictions_{args.split}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not sample:
        # still write header-only useful for tooling
        print(f"No errors found on {args.split} (or empty split).")
    fieldnames = list(sample[0].keys()) if sample else [
        "item_id",
        "error_bucket",
        "true_sleeve",
        "pred_sleeve",
        "true_media",
        "pred_media",
        "text_clean",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sample)

    print(
        f"Wrote {len(sample)} / {len(errors)} error rows to {out_path} "
        f"(split={args.split}, stratify={args.stratify})"
    )


if __name__ == "__main__":
    main()
