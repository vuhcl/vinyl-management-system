from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import types
from pathlib import Path

from grader.src.eval.label_audit_backend import (
    commit_queue_to_label_patches,
    compute_disagreement_flags,
    ensure_db,
    parse_llm_json,
    validate_decision,
)
from grader.src.eval import label_audit_run_llm

label_audit_run_llm_lib = importlib.import_module(
    "grader.src.eval.label_audit_run_llm.lib"
)


def test_is_quota_exhausted_error_detects_common_signals() -> None:
    assert label_audit_run_llm._is_quota_exhausted_error(
        Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
    )
    assert label_audit_run_llm._is_quota_exhausted_error(
        Exception("Too many requests; rate limit reached")
    )
    assert not label_audit_run_llm._is_quota_exhausted_error(
        Exception("JSON parse error")
    )


def test_is_ollama_transient_error_detects_timeouts_and_runner_failures() -> None:
    assert label_audit_run_llm._is_ollama_transient_error(Exception("Request timed out."))
    assert label_audit_run_llm._is_ollama_transient_error(
        Exception(
            "Error code: 500 - {'error': {'message': 'llama runner process has terminated: ...'}}"
        )
    )
    assert not label_audit_run_llm._is_ollama_transient_error(
        Exception("JSON parse error")
    )


def test_parse_llm_json_handles_fenced_block() -> None:
    raw = """```json
{"llm_verdict":"Near Mint","llm_confidence":0.81,
"reason_code":"supports_higher","llm_abstain":false}
```"""
    obj = parse_llm_json(raw)
    assert obj["llm_verdict"] == "Near Mint"


def test_parse_llm_json_handles_prose_wrapped_json() -> None:
    raw = (
        "Here is the result.\n"
        '{"llm_verdict":"Very Good","llm_confidence":0.62,'
        '"reason_code":"mixed_signals","llm_abstain":false}\n'
        "Thanks!"
    )
    obj = parse_llm_json(raw)
    assert obj["llm_verdict"] == "Very Good"


def test_parse_llm_json_handles_trailing_commas() -> None:
    raw = """```json
{
  "llm_verdict": "Near Mint",
  "llm_confidence": 0.91,
  "reason_code": "supports_higher",
  "llm_abstain": false,
}
```"""
    obj = parse_llm_json(raw)
    assert obj["reason_code"] == "supports_higher"


def test_parse_llm_json_handles_smart_quotes() -> None:
    raw = (
        '{“llm_verdict”:“Near Mint”,“llm_confidence”:0.8,'
        '“reason_code”:“supports_higher”,“llm_abstain”:false}'
    )
    obj = parse_llm_json(raw)
    assert obj["llm_verdict"] == "Near Mint"


def test_validate_decision_checks_allowed() -> None:
    obj = {
        "llm_verdict": "Near Mint",
        "llm_confidence": 0.5,
        "reason_code": "mixed_signals",
        "llm_abstain": False,
    }
    d = validate_decision(obj, ["Mint", "Near Mint"])
    assert d.verdict == "Near Mint"


def test_compute_disagreement_flags() -> None:
    flags = compute_disagreement_flags(
        assigned_label="Very Good Plus",
        model_pred_label="Near Mint",
        llm_verdict="Near Mint",
    )
    assert flags["disagree_assigned_vs_model"] == 1
    assert flags["disagree_llm_vs_assigned"] == 1
    assert flags["disagree_llm_vs_model"] == 0


def test_commit_queue_to_label_patches_groups_media_and_sleeve(
    tmp_path: Path,
) -> None:
    db = tmp_path / "q.sqlite"
    ensure_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO queue(
                source,item_id,split,target,assigned_label,llm_verdict,human_action,final_label
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                "discogs",
                "123",
                "train",
                "media",
                "Very Good",
                "Near Mint",
                "accept_llm",
                "",
            ),
        )
        conn.execute(
            """
            INSERT INTO queue(
                source,item_id,split,target,assigned_label,llm_verdict,human_action,final_label
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                "discogs",
                "123",
                "train",
                "sleeve",
                "Very Good",
                "Very Good Plus",
                "accept_llm",
                "",
            ),
        )
        conn.commit()
    patches = tmp_path / "label_patches.jsonl"
    temp_csv = tmp_path / "preview.csv"
    out = commit_queue_to_label_patches(
        db_path=db,
        label_patches_path=patches,
        temp_csv_path=temp_csv,
    )
    assert out["grouped_rows"] == 1
    lines = patches.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["media_label"] == "Near Mint"
    assert rec["sleeve_label"] == "Very Good Plus"


def test_run_llm_respects_targets_filter(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "queue.sqlite"
    ensure_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO queue(
                source,item_id,split,target,text,assigned_label,model_pred_label,
                cleanlab_label_issue,llm_status
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                "discogs",
                "123",
                "train",
                "sleeve",
                "Light ringwear, glossy vinyl.",
                "Very Good",
                "Near Mint",
                1,
                "",
            ),
        )
        conn.execute(
            """
            INSERT INTO queue(
                source,item_id,split,target,text,assigned_label,model_pred_label,
                cleanlab_label_issue,llm_status
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                "discogs",
                "123",
                "train",
                "media",
                "Light ringwear, glossy vinyl.",
                "Very Good",
                "Near Mint",
                1,
                "",
            ),
        )
        conn.commit()

    cfg = tmp_path / "grader.yaml"
    cfg.write_text("project: test\n", encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=test\nGEMINI_MODEL=gemini-test\n", encoding="utf-8")
    raw_dir = tmp_path / "raw"

    def _fake_guides(_path: Path, target: str) -> dict[str, list]:
        if target == "media":
            return {
                "allowed": ["Very Good", "Near Mint"],
                "descriptions": [("Near Mint", "NM desc"), ("Very Good", "VG desc")],
            }
        return {
            "allowed": ["Very Good", "Very Good Plus"],
            "descriptions": [
                ("Very Good Plus", "VG+ desc"),
                ("Very Good", "VG desc"),
            ],
        }

    monkeypatch.setattr(
        label_audit_run_llm_lib, "load_guideline_prompt_bits", _fake_guides
    )
    _gemini_payload = json.dumps(
        {
            "media": {
                "llm_verdict": "Near Mint",
                "llm_confidence": 0.9,
                "reason_code": "supports_higher",
                "llm_abstain": False,
            },
            "sleeve": {
                "llm_verdict": "Very Good Plus",
                "llm_confidence": 0.8,
                "reason_code": "supports_higher",
                "llm_abstain": False,
            },
        }
    )

    monkeypatch.setattr(
        label_audit_run_llm_lib,
        "_query_gemini",
        lambda _messages, model_id: (_gemini_payload, model_id),
    )
    monkeypatch.setattr(label_audit_run_llm.time, "sleep", lambda _seconds: None)
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_audit_run_llm",
            "--config",
            str(cfg),
            "--guidelines",
            str(cfg),
            "--db",
            str(db),
            "--raw-dir",
            str(raw_dir),
            "--gating-pass",
            "1",
                "--provider",
                "gemini",
            "--splits",
            "train",
            "--targets",
            "sleeve",
            "--env-file",
            str(env),
        ],
    )

    assert label_audit_run_llm.main() == 0

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        sleeve = conn.execute(
            "SELECT llm_status,llm_verdict FROM queue WHERE target='sleeve'"
        ).fetchone()
        media = conn.execute(
            "SELECT llm_status,llm_verdict FROM queue WHERE target='media'"
        ).fetchone()

    assert sleeve is not None
    assert media is not None
    assert sleeve["llm_status"] == "ok"
    assert sleeve["llm_verdict"] == "Very Good Plus"
    assert media["llm_status"] == ""
    assert media["llm_verdict"] == ""
