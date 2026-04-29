from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from grader.src.eval.label_audit_backend import ensure_db
from grader.src.eval import label_audit_calibrate_policy


def _insert_row(
    conn: sqlite3.Connection,
    *,
    queue_row_id: int,
    target: str,
    assigned_label: str,
    model_pred_label: str,
    model_pred_proba_max: float,
    model_proba_assigned: float,
    llm_verdict: str,
    llm_confidence: float,
    human_action: str = "",
    final_label: str = "",
    llm_abstain: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO queue(
            queue_row_id,source,item_id,split,target,text,assigned_label,model_pred_label,
            model_pred_proba_max,model_proba_assigned,llm_verdict,llm_confidence,llm_abstain,
            human_action,final_label
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            queue_row_id,
            "discogs",
            f"{target}:{queue_row_id}",
            "train",
            target,
            "test text",
            assigned_label,
            model_pred_label,
            float(model_pred_proba_max),
            float(model_proba_assigned),
            llm_verdict,
            float(llm_confidence),
            int(llm_abstain),
            human_action,
            final_label,
        ),
    )


def test_ensure_db_adds_auto_decision_columns_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "q.sqlite"
    ensure_db(db)
    ensure_db(db)
    with sqlite3.connect(db) as conn:
        cols = {
            str(r[1])
            for r in conn.execute("PRAGMA table_info(queue)").fetchall()
        }
    assert "auto_decision" in cols
    assert "auto_final_label" in cols
    assert "auto_decision_score" in cols
    assert "auto_decision_reason" in cols
    assert "auto_decision_source" in cols
    assert "auto_policy_version" in cols


def test_calibrate_apply_and_clear_policy_version(
    tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "q.sqlite"
    ensure_db(db)
    with sqlite3.connect(db) as conn:
        # reviewed sample (small but forced)
        _insert_row(
            conn,
            queue_row_id=1,
            target="media",
            assigned_label="Very Good",
            model_pred_label="Near Mint",
            model_pred_proba_max=0.96,
            model_proba_assigned=0.20,
            llm_verdict="Very Good Plus",
            llm_confidence=0.55,
            human_action="manual_set",
            final_label="Near Mint",
        )
        _insert_row(
            conn,
            queue_row_id=2,
            target="sleeve",
            assigned_label="Very Good",
            model_pred_label="Very Good",
            model_pred_proba_max=0.30,
            model_proba_assigned=0.92,
            llm_verdict="Very Good",
            llm_confidence=0.60,
            human_action="keep_assigned",
            final_label="",
        )
        _insert_row(
            conn,
            queue_row_id=3,
            target="media",
            assigned_label="Very Good",
            model_pred_label="Near Mint",
            model_pred_proba_max=0.97,
            model_proba_assigned=0.10,
            llm_verdict="Very Good",
            llm_confidence=0.40,
            human_action="",
            final_label="",
        )
        conn.commit()

    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_audit_calibrate_policy",
            "--db",
            str(db),
            "--report-path",
            str(report_path),
            "--policy-version",
            "p-test",
            "--apply",
            "--force",
        ],
    )
    assert label_audit_calibrate_policy.main() == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["policy_version"] == "p-test"
    assert "thresholds" in report
    assert "holdout" in report

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT auto_decision,auto_final_label,auto_decision_source,
                   auto_policy_version
            FROM queue WHERE queue_row_id=3
            """
        ).fetchone()
    assert row is not None
    assert str(row["auto_policy_version"]) == "p-test"
    assert str(row["auto_decision"]) in {"proposed_grade", "needs_review"}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_audit_calibrate_policy",
            "--db",
            str(db),
            "--clear-policy-version",
            "p-test",
        ],
    )
    assert label_audit_calibrate_policy.main() == 0
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row2 = conn.execute(
            """
            SELECT auto_decision,auto_policy_version
            FROM queue
            WHERE queue_row_id=3
            """
        ).fetchone()
    assert row2 is not None
    assert str(row2["auto_decision"] or "") == ""
    assert str(row2["auto_policy_version"] or "") == ""
