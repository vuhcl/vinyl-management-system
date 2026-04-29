from __future__ import annotations

import sqlite3
from pathlib import Path

from grader.src.eval.label_audit_critic import (
    CriticDecision,
    CriticExample,
    derive_gold_label,
    parse_critic_decision,
    retrieve_examples,
    to_auto_decision,
)
from grader.src.eval.label_audit_backend import ensure_db


def test_derive_gold_label_from_human_actions() -> None:
    r1 = {"human_action": "accept_llm", "llm_verdict": "Very Good"}
    r2 = {"human_action": "keep_assigned", "assigned_label": "Good"}
    r3 = {"human_action": "manual_set", "final_label": "Near Mint"}
    r4 = {"human_action": "auto_apply", "final_label": "Generic"}
    assert derive_gold_label(r1) == ("Very Good", "accept_llm")
    assert derive_gold_label(r2) == ("Good", "keep_assigned")
    assert derive_gold_label(r3) == ("Near Mint", "manual_set")
    assert derive_gold_label(r4) == ("Generic", "auto_apply")


def test_retrieve_examples_prefers_lexical_overlap() -> None:
    bank = [
        CriticExample(
            target="media",
            text="record has crackle and some noise but plays through",
            assigned_label="Very Good",
            model_pred_label="Good",
            llm_verdict="Very Good",
            gold_label="Very Good",
            gold_source="manual_set",
            split="train",
            queue_row_id=1,
        ),
        CriticExample(
            target="media",
            text="sleeve seam split heavy ring wear",
            assigned_label="Good",
            model_pred_label="Good",
            llm_verdict="Good",
            gold_label="Good",
            gold_source="manual_set",
            split="train",
            queue_row_id=2,
        ),
    ]
    got = retrieve_examples(
        bank=bank,
        target="media",
        text="light crackle with noise, still plays",
        split_exclude="val",
        row_id_exclude=99,
        k=1,
    )
    assert len(got) == 1
    assert got[0].queue_row_id == 1


def test_parse_critic_decision_validates_labels() -> None:
    allowed = ["Very Good", "Good"]
    obj = parse_critic_decision(
        '{"proposed_grade":"Very Good","confidence":0.94,"needs_review":false,"reason":"matched examples"}',
        allowed,
    )
    assert obj.proposed_grade == "Very Good"
    assert obj.needs_review is False
    assert obj.confidence == 0.94


def test_parse_critic_decision_rejects_invalid_label() -> None:
    allowed = ["Very Good", "Good"]
    try:
        parse_critic_decision(
            '{"proposed_grade":"Near Mint","confidence":0.99,"needs_review":false,"reason":"bad"}',
            allowed,
        )
    except ValueError as exc:
        assert "Invalid proposed_grade" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for invalid label.")


def test_to_auto_decision_safety_fallback() -> None:
    d = CriticDecision(
        proposed_grade="Very Good",
        confidence=0.82,
        needs_review=False,
        reason="uncertain",
    )
    auto = to_auto_decision(d, min_confidence=0.9)
    assert auto[0] == "needs_review"
    assert auto[1] == ""


def test_queue_write_fields_for_critic_decisions(tmp_path: Path) -> None:
    db = tmp_path / "queue.sqlite"
    ensure_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO queue
            (split,source,item_id,target,text,assigned_label,model_pred_label,llm_status)
            VALUES ('train','discogs','x1','media','txt','Very Good','Good','ok')
            """
        )
        row_id = int(conn.execute("SELECT queue_row_id FROM queue").fetchone()[0])
        conn.execute(
            """
            UPDATE queue
            SET auto_decision=?, auto_final_label=?, auto_decision_score=?,
                auto_decision_reason=?, auto_decision_source=?, auto_policy_version=?
            WHERE queue_row_id=?
            """,
            ("proposed_grade", "Very Good", 0.97, "critic ok", "llm_critic", "critic-v1", row_id),
        )
        row = conn.execute(
            "SELECT auto_decision,auto_final_label,auto_decision_source FROM queue WHERE queue_row_id=?",
            (row_id,),
        ).fetchone()
    assert row[0] == "proposed_grade"
    assert row[1] == "Very Good"
    assert row[2] == "llm_critic"
