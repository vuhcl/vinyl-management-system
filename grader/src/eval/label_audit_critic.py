from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from grader.src.eval.label_audit_backend import parse_llm_json


def derive_gold_label(row: sqlite3.Row | dict[str, Any]) -> tuple[str, str]:
    action = str(row["human_action"] if isinstance(row, sqlite3.Row) else row.get("human_action", "") or "").strip()
    if action == "accept_llm":
        label = str(row["llm_verdict"] if isinstance(row, sqlite3.Row) else row.get("llm_verdict", "") or "").strip()
        return label, "accept_llm"
    if action == "keep_assigned":
        label = str(row["assigned_label"] if isinstance(row, sqlite3.Row) else row.get("assigned_label", "") or "").strip()
        return label, "keep_assigned"
    if action in {"manual_set", "auto_apply"}:
        label = str(row["final_label"] if isinstance(row, sqlite3.Row) else row.get("final_label", "") or "").strip()
        return label, action
    return "", ""


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2}


@dataclass
class CriticExample:
    target: str
    text: str
    assigned_label: str
    model_pred_label: str
    llm_verdict: str
    gold_label: str
    gold_source: str
    split: str
    queue_row_id: int


def load_human_gold_examples(
    conn: sqlite3.Connection,
    *,
    include_auto_apply: bool = False,
) -> list[CriticExample]:
    actions = ["accept_llm", "keep_assigned", "manual_set"]
    if include_auto_apply:
        actions.append("auto_apply")
    placeholders = ",".join("?" for _ in actions)
    rows = conn.execute(
        f"""
        SELECT queue_row_id,target,text,split,assigned_label,model_pred_label,
               llm_verdict,human_action,final_label
        FROM queue
        WHERE COALESCE(human_action, '') IN ({placeholders})
        """,
        tuple(actions),
    ).fetchall()
    out: list[CriticExample] = []
    for row in rows:
        gold_label, gold_source = derive_gold_label(row)
        if not gold_label:
            continue
        out.append(
            CriticExample(
                target=str(row["target"] or "").strip().lower(),
                text=str(row["text"] or ""),
                assigned_label=str(row["assigned_label"] or "").strip(),
                model_pred_label=str(row["model_pred_label"] or "").strip(),
                llm_verdict=str(row["llm_verdict"] or "").strip(),
                gold_label=gold_label,
                gold_source=gold_source,
                split=str(row["split"] or "").strip(),
                queue_row_id=int(row["queue_row_id"] or 0),
            )
        )
    return out


def retrieve_examples(
    *,
    bank: list[CriticExample],
    target: str,
    text: str,
    split_exclude: str | None,
    row_id_exclude: int | None,
    k: int,
) -> list[CriticExample]:
    if k <= 0:
        return []
    q_tokens = _tokenize(text)
    scored: list[tuple[float, CriticExample]] = []
    for ex in bank:
        if ex.target != target:
            continue
        if split_exclude and ex.split == split_exclude and row_id_exclude == ex.queue_row_id:
            continue
        ex_tokens = _tokenize(ex.text)
        if not ex_tokens:
            continue
        inter = len(q_tokens & ex_tokens)
        union = len(q_tokens | ex_tokens) or 1
        jacc = inter / union
        if jacc <= 0:
            continue
        scored.append((jacc, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _s, ex in scored[:k]]


def build_critic_messages(
    *,
    target: str,
    text: str,
    assigned_label: str,
    model_pred_label: str,
    llm_verdict: str,
    llm_confidence: float,
    allowed_labels: list[str],
    examples: list[CriticExample],
) -> list[dict[str, str]]:
    examples_payload = [
        {
            "text": ex.text[:400],
            "assigned_label": ex.assigned_label,
            "model_pred_label": ex.model_pred_label,
            "llm_verdict": ex.llm_verdict,
            "gold_label": ex.gold_label,
            "gold_source": ex.gold_source,
        }
        for ex in examples
    ]
    system = (
        "You are a strict critic that reviews vinyl label predictions.\n"
        "Decide whether to propose a final grade or route to needs_review.\n"
        "Return JSON only with keys exactly: proposed_grade, confidence, needs_review, reason.\n"
        "If uncertain, set needs_review=true and proposed_grade=''.\n"
        "Use only allowed labels for proposed_grade when needs_review=false.\n"
    )
    user = {
        "target": target,
        "allowed_labels": allowed_labels,
        "listing_text": text[:4000],
        "candidates": {
            "assigned_label": assigned_label,
            "model_pred_label": model_pred_label,
            "llm_verdict": llm_verdict,
            "llm_confidence": float(llm_confidence),
        },
        "manual_examples": examples_payload,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


@dataclass
class CriticDecision:
    proposed_grade: str
    confidence: float
    needs_review: bool
    reason: str


def to_auto_decision(
    decision: CriticDecision, *, min_confidence: float
) -> tuple[str, str, float, str]:
    if (
        (not decision.needs_review)
        and decision.proposed_grade
        and decision.confidence >= float(min_confidence)
    ):
        return (
            "proposed_grade",
            decision.proposed_grade,
            float(decision.confidence),
            decision.reason,
        )
    return ("needs_review", "", float(decision.confidence), decision.reason)


def parse_critic_decision(raw_text: str, allowed_labels: list[str]) -> CriticDecision:
    obj = parse_llm_json(raw_text)
    needs_review = bool(obj.get("needs_review", True))
    proposed = str(obj.get("proposed_grade", "") or "").strip()
    reason = str(obj.get("reason", "") or "").strip()
    conf = float(obj.get("confidence", 0.0) or 0.0)
    conf = max(0.0, min(1.0, conf))
    if needs_review:
        proposed = ""
    elif proposed not in allowed_labels:
        raise ValueError(
            f"Invalid proposed_grade={proposed!r}; allowed={allowed_labels!r}"
        )
    return CriticDecision(
        proposed_grade=proposed,
        confidence=conf,
        needs_review=needs_review,
        reason=reason[:500],
    )
