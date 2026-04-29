from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from grader.src.eval.label_audit_critic import (
    build_critic_messages,
    derive_gold_label,
    load_human_gold_examples,
    parse_critic_decision,
    retrieve_examples,
)
from grader.src.eval.label_audit_backend import load_guideline_prompt_bits
from grader.src.eval.label_audit_run_llm import _query_provider


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    proposed = [r for r in rows if r["proposed"]]
    correct = [r for r in proposed if r["correct"]]
    total = len(rows)
    return {
        "precision": (len(correct) / len(proposed)) if proposed else 0.0,
        "coverage": (len(proposed) / total) if total else 0.0,
        "proposed": len(proposed),
        "correct": len(correct),
        "total": total,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate first-pass vs critic pass.")
    p.add_argument("--db", default="grader/reports/label_audit_queue.sqlite")
    p.add_argument("--provider", choices=["gemini", "openrouter", "ollama"], default="ollama")
    p.add_argument("--model-id", default="")
    p.add_argument("--critic-min-confidence", type=float, default=0.90)
    p.add_argument("--critic-k-examples", type=int, default=5)
    p.add_argument("--holdout-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--env-file", default=".env")
    p.add_argument("--report-path", default="grader/reports/critic_eval_report.json")
    p.add_argument("--guidelines", default="grader/configs/grading_guidelines.yaml")
    args = p.parse_args()
    load_dotenv(args.env_file)
    media_allowed = load_guideline_prompt_bits(Path(args.guidelines), "media")["allowed"]
    sleeve_allowed = load_guideline_prompt_bits(Path(args.guidelines), "sleeve")["allowed"]

    with sqlite3.connect(Path(args.db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT queue_row_id,target,split,text,assigned_label,model_pred_label,llm_verdict,
                   llm_confidence,human_action,final_label
            FROM queue
            WHERE COALESCE(human_action, '') <> ''
            """
        ).fetchall()
        candidates = []
        for r in rows:
            gold, src = derive_gold_label(r)
            if not gold:
                continue
            candidates.append((r, gold, src))
        if len(candidates) < 20:
            raise ValueError("Need at least 20 reviewed rows for holdout evaluation.")
        random.seed(int(args.seed))
        random.shuffle(candidates)
        cut = max(1, int(round(len(candidates) * (1.0 - float(args.holdout_ratio)))))
        train = candidates[:cut]
        holdout = candidates[cut:]
        bank = load_human_gold_examples(conn, include_auto_apply=False)
        train_ids = {int(r["queue_row_id"]) for r, _g, _s in train}

    first_pass_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    by_target: dict[str, dict[str, Any]] = {"media": {}, "sleeve": {}}

    for r, gold, _src in holdout:
        target = str(r["target"])
        llm_verdict = str(r["llm_verdict"] or "").strip()
        first_proposed = bool(llm_verdict)
        first_pass_rows.append(
            {"target": target, "proposed": first_proposed, "correct": llm_verdict == gold}
        )
        allowed_labels = media_allowed if target == "media" else sleeve_allowed
        holdout_id = int(r["queue_row_id"])
        examples = retrieve_examples(
            bank=[ex for ex in bank if ex.queue_row_id in train_ids],
            target=target,
            text=str(r["text"] or ""),
            split_exclude=str(r["split"] or ""),
            row_id_exclude=holdout_id,
            k=int(args.critic_k_examples),
        )
        msgs = build_critic_messages(
            target=target,
            text=str(r["text"] or ""),
            assigned_label=str(r["assigned_label"] or ""),
            model_pred_label=str(r["model_pred_label"] or ""),
            llm_verdict=llm_verdict,
            llm_confidence=float(r["llm_confidence"] or 0.0),
            allowed_labels=allowed_labels,
            examples=examples,
        )
        try:
            raw, _used = _query_provider(
                provider=args.provider,
                messages=msgs,
                model_id=args.model_id.strip() or "",
            )
            crit = parse_critic_decision(raw, allowed_labels)
            proposed = (
                (not crit.needs_review)
                and bool(crit.proposed_grade)
                and float(crit.confidence) >= float(args.critic_min_confidence)
            )
            final_grade = crit.proposed_grade if proposed else ""
        except Exception:
            proposed = False
            final_grade = ""
        critic_rows.append({"target": target, "proposed": proposed, "correct": final_grade == gold})

    report = {
        "reviewed_total": len(candidates),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "first_pass": _metrics(first_pass_rows),
        "critic_pass": _metrics(critic_rows),
        "split_hygiene": {
            "retrieval_from_train_only": True,
            "no_self_retrieval": True,
        },
    }
    for t in ("media", "sleeve"):
        report[f"first_pass_{t}"] = _metrics([r for r in first_pass_rows if r["target"] == t])
        report[f"critic_pass_{t}"] = _metrics([r for r in critic_rows if r["target"] == t])

    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "critic_eval_report", "path": str(out_path), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
