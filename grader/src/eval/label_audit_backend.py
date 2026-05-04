from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grader.src.config_io import load_yaml_mapping
from grader.src.data.label_patches import append_csv_to_label_patches
from grader.src.eval.label_audit_constants import (
    COMMIT_QUEUE_HUMAN_ACTIONS,
    human_action_sql_in_list,
)


VALID_SPLITS = ("train", "val", "test")
VALID_TARGETS = ("sleeve", "media")
REASON_CODES = (
    "supports_higher",
    "supports_lower",
    "insufficient_evidence",
    "mixed_signals",
    "ambiguous_target",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queue (
                queue_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                item_id TEXT NOT NULL,
                split TEXT NOT NULL,
                target TEXT NOT NULL,
                text TEXT DEFAULT '',
                text_clean TEXT DEFAULT '',
                assigned_label TEXT DEFAULT '',
                sibling_label TEXT DEFAULT '',
                model_pred_label TEXT DEFAULT '',
                model_pred_proba_max REAL,
                model_proba_assigned REAL,
                cleanlab_self_confidence REAL,
                cleanlab_label_issue INTEGER DEFAULT 0,
                label_confidence REAL,
                llm_status TEXT DEFAULT '',
                llm_verdict TEXT DEFAULT '',
                llm_confidence REAL,
                reason_code TEXT DEFAULT '',
                llm_abstain INTEGER DEFAULT 0,
                llm_model_id TEXT DEFAULT '',
                prompt_version TEXT DEFAULT '',
                audit_run_id TEXT DEFAULT '',
                raw_response_path TEXT DEFAULT '',
                response_cache_key TEXT DEFAULT '',
                llm_error TEXT DEFAULT '',
                disagree_assigned_vs_model INTEGER DEFAULT 0,
                disagree_llm_vs_assigned INTEGER DEFAULT 0,
                disagree_llm_vs_model INTEGER DEFAULT 0,
                low_cleanlab_confidence INTEGER DEFAULT 0,
                human_action TEXT DEFAULT '',
                final_label TEXT DEFAULT '',
                auto_decision TEXT DEFAULT '',
                auto_final_label TEXT DEFAULT '',
                auto_decision_score REAL,
                auto_decision_reason TEXT DEFAULT '',
                auto_decision_source TEXT DEFAULT '',
                auto_policy_version TEXT DEFAULT '',
                updated_at TEXT DEFAULT '',
                UNIQUE(source, item_id, split, target)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_queue_columns(
            conn,
            {
                "auto_decision": "TEXT DEFAULT ''",
                "auto_final_label": "TEXT DEFAULT ''",
                "auto_decision_score": "REAL",
                "auto_decision_reason": "TEXT DEFAULT ''",
                "auto_decision_source": "TEXT DEFAULT ''",
                "auto_policy_version": "TEXT DEFAULT ''",
            },
        )
        conn.commit()


def _ensure_queue_columns(
    conn: sqlite3.Connection, columns: dict[str, str]
) -> None:
    existing = {
        str(row[1]).strip()
        for row in conn.execute("PRAGMA table_info(queue)").fetchall()
        if row and len(row) > 1
    }
    for col_name, col_decl in columns.items():
        if col_name in existing:
            continue
        conn.execute(f"ALTER TABLE queue ADD COLUMN {col_name} {col_decl}")


def _load_split_index_from_jsonl(
    splits_dir: Path,
) -> dict[tuple[str, str, str], dict[str, str]]:
    idx: dict[tuple[str, str, str], dict[str, str]] = {}
    for split in VALID_SPLITS:
        p = splits_dir / f"{split}.jsonl"
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (
                    split,
                    str(rec.get("source", "")).strip(),
                    str(rec.get("item_id", "")).strip(),
                )
                idx[key] = {
                    "text": str(rec.get("text", "")),
                    "text_clean": str(rec.get("text_clean", "")),
                    "sleeve_label": str(rec.get("sleeve_label", "")),
                    "media_label": str(rec.get("media_label", "")),
                }
    return idx


def infer_split_from_filename(path: Path, default_split: str = "train") -> str:
    name = path.name.lower()
    for split in VALID_SPLITS:
        if f"_{split}_" in name or name.endswith(f"_{split}.csv"):
            return split
    return default_split


def build_queue_from_cleanlab_csvs(
    *,
    db_path: Path,
    csv_paths: list[Path],
    splits_dir: Path,
    default_split: str = "train",
) -> dict[str, int]:
    ensure_db(db_path)
    split_index = _load_split_index_from_jsonl(splits_dir)
    inserted = 0
    updated = 0

    with sqlite3.connect(db_path) as conn:
        for csv_path in csv_paths:
            with csv_path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    target = str(row.get("target", "")).strip()
                    if target not in VALID_TARGETS:
                        continue
                    split_raw = str(row.get("split", "")).strip()
                    split = split_raw or infer_split_from_filename(
                        csv_path, default_split=default_split
                    )
                    if split not in VALID_SPLITS:
                        continue
                    source = str(row.get("source", "")).strip()
                    item_id = str(row.get("item_id", "")).strip()
                    if not source or not item_id:
                        continue
                    txt = split_index.get((split, source, item_id), {})
                    assigned = str(
                        row.get(f"{target}_label", "")
                        or row.get("assigned_label", "")
                    ).strip()
                    sibling = str(
                        row.get("media_label", "")
                        if target == "sleeve"
                        else row.get("sleeve_label", "")
                    ).strip()
                    model_pred = str(
                        row.get("model_pred_label", "")
                        or row.get("oof_pred_label", "")
                    ).strip()
                    cleanlab_issue = (
                        str(row.get("cleanlab_label_issue", ""))
                        .strip()
                        .lower()
                        in ("1", "true", "yes")
                    )
                    cleanlab_sc = _to_float(row.get("cleanlab_self_confidence"))
                    pred_max = _to_float(
                        row.get("model_pred_proba_max")
                        or row.get("oof_pred_proba_max")
                    )
                    pred_asg = _to_float(row.get("model_proba_assigned"))
                    lbl_conf = _to_float(row.get("label_confidence"))
                    low_cl = (
                        1
                        if cleanlab_sc is not None and cleanlab_sc <= 0.20
                        else 0
                    )
                    d_asg_model = (
                        1
                        if assigned and model_pred and assigned != model_pred
                        else 0
                    )
                    now = utc_now_iso()
                    existed = conn.execute(
                        """
                        SELECT 1
                        FROM queue
                        WHERE source=? AND item_id=? AND split=? AND target=?
                        """,
                        (source, item_id, split, target),
                    ).fetchone()
                    cur = conn.execute(
                        """
                        INSERT INTO queue(
                            source,item_id,split,target,text,text_clean,assigned_label,sibling_label,
                            model_pred_label,model_pred_proba_max,model_proba_assigned,
                            cleanlab_self_confidence,cleanlab_label_issue,label_confidence,
                            disagree_assigned_vs_model,low_cleanlab_confidence,updated_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(source,item_id,split,target) DO UPDATE SET
                            text=excluded.text,
                            text_clean=excluded.text_clean,
                            assigned_label=excluded.assigned_label,
                            sibling_label=excluded.sibling_label,
                            model_pred_label=excluded.model_pred_label,
                            model_pred_proba_max=excluded.model_pred_proba_max,
                            model_proba_assigned=excluded.model_proba_assigned,
                            cleanlab_self_confidence=excluded.cleanlab_self_confidence,
                            cleanlab_label_issue=excluded.cleanlab_label_issue,
                            label_confidence=excluded.label_confidence,
                            disagree_assigned_vs_model=excluded.disagree_assigned_vs_model,
                            low_cleanlab_confidence=excluded.low_cleanlab_confidence,
                            updated_at=excluded.updated_at
                        """,
                        (
                            source,
                            item_id,
                            split,
                            target,
                            str(txt.get("text", "")),
                            str(txt.get("text_clean", "")),
                            assigned,
                            sibling,
                            model_pred,
                            pred_max,
                            pred_asg,
                            cleanlab_sc,
                            int(cleanlab_issue),
                            lbl_conf,
                            d_asg_model,
                            low_cl,
                            now,
                        ),
                    )
                    if existed:
                        updated += 1
                    elif cur.rowcount > 0:
                        inserted += 1
        conn.commit()
    return {
        "inserted_or_updated": inserted + updated,
        "inserted": inserted,
        "updated": updated,
    }


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_guideline_prompt_bits(
    guidelines_path: Path, target: str
) -> dict[str, Any]:
    cfg = load_yaml_mapping(guidelines_path)
    if target == "media":
        allowed = [str(x) for x in (cfg.get("media_grades") or [])]
    else:
        allowed = [str(x) for x in (cfg.get("sleeve_grades") or [])]
    grade_defs = cfg.get("grades") or {}
    descs: list[tuple[str, str]] = []
    rubric_lines: list[str] = []
    signal_keys = (
        f"supporting_signals_{target}",
        "supporting_signals",
        f"forbidden_signals_{target}",
        f"hard_signals_strict_{target}",
        f"hard_signals_cosignal_{target}",
    )
    for grade_name, info in grade_defs.items():
        if not isinstance(info, dict):
            continue
        applies = [str(x) for x in (info.get("applies_to") or [])]
        if applies and target not in applies:
            continue
        if grade_name not in allowed:
            continue
        desc = str(info.get("description", "")).strip()
        if desc:
            descs.append((str(grade_name), desc))
            rubric_lines.append(f"{grade_name}: {desc}")
        for key in signal_keys:
            vals = info.get(key)
            if not isinstance(vals, list):
                continue
            cleaned = [str(v).strip() for v in vals if str(v).strip()]
            if not cleaned:
                continue
            # Keep prompt length bounded while still conveying broader rubric intent.
            rubric_lines.append(f"{grade_name} {key}: {', '.join(cleaned[:25])}")
    rubric_text = "\n".join(rubric_lines)
    return {"allowed": allowed, "descriptions": descs, "rubric_text": rubric_text}


def prompt_version_hash(
    *,
    target: str,
    model_id: str,
    guideline_descriptions: list[tuple[str, str]],
    allowed_labels: list[str],
) -> str:
    payload = json.dumps(
        {
            "target": target,
            "model_id": model_id,
            "allowed": allowed_labels,
            "descriptions": guideline_descriptions,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_prompt_messages(
    *,
    row: dict[str, Any],
    target: str,
    allowed_labels: list[str],
    grade_descriptions: list[tuple[str, str]],
) -> list[dict[str, str]]:
    defs = "\n".join(f"- {g}: {d}" for g, d in grade_descriptions)
    system = (
        "You audit vinyl listing labels.\n"
        f"Target: {target}.\n"
        f"Allowed labels (exact strings): {allowed_labels}\n"
        "Return JSON only with keys: "
        "llm_verdict, llm_confidence, reason_code, llm_abstain.\n"
        f"reason_code must be one of {list(REASON_CODES)}.\n"
        "Use listing text evidence first. If insufficient evidence, set llm_abstain=true.\n"
    )
    user_payload = {
        "target": target,
        "source": row["source"],
        "item_id": row["item_id"],
        "text": row.get("text", "")[:4000],
        "assigned_label": row.get("assigned_label", ""),
        "model_pred_label": row.get("model_pred_label", ""),
        "cleanlab_self_confidence": row.get("cleanlab_self_confidence"),
        "cleanlab_label_issue": bool(row.get("cleanlab_label_issue")),
        "grade_definitions": defs,
    }
    user_content = json.dumps(user_payload, ensure_ascii=False)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_llm_json_text(text: str) -> str:
    # Strip markdown fences if model ignored "JSON only".
    t = text.strip()
    if t.startswith("```"):
        m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n([\s\S]*?)\n```$", t)
        if m:
            t = m.group(1).strip()
        else:
            t = t.strip("`").strip()
    # Normalize smart quotes and remove BOM.
    t = (
        t.replace("\ufeff", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    # Keep only the first balanced JSON object when prose wraps it.
    obj_candidate = _extract_balanced_json_object(t)
    if obj_candidate:
        t = obj_candidate
    # Remove trailing commas before } or ] (common model slip).
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def parse_llm_json(raw_text: str) -> dict[str, Any]:
    text = _normalize_llm_json_text(raw_text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: slice from first "{" to last "}".
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(obj, dict):
        raise ValueError("LLM response is not a JSON object.")
    return obj


@dataclass
class LLMDecision:
    verdict: str
    confidence: float
    reason_code: str
    abstain: bool


def validate_decision(
    obj: dict[str, Any], allowed_labels: list[str]
) -> LLMDecision:
    abstain = bool(obj.get("llm_abstain", False))
    reason_code = str(obj.get("reason_code", "")).strip()
    if reason_code not in REASON_CODES:
        raise ValueError(f"Invalid reason_code={reason_code!r}.")
    conf = float(obj.get("llm_confidence", 0.0))
    conf = max(0.0, min(1.0, conf))
    verdict = str(obj.get("llm_verdict", "")).strip()
    if not abstain and verdict not in allowed_labels:
        raise ValueError(
            f"Invalid llm_verdict={verdict!r}; allowed={allowed_labels!r}"
        )
    return LLMDecision(
        verdict=verdict,
        confidence=conf,
        reason_code=reason_code,
        abstain=abstain,
    )


def compute_disagreement_flags(
    *,
    assigned_label: str,
    model_pred_label: str,
    llm_verdict: str,
) -> dict[str, int]:
    return {
        "disagree_assigned_vs_model": int(
            bool(
                assigned_label
                and model_pred_label
                and assigned_label != model_pred_label
            )
        ),
        "disagree_llm_vs_assigned": int(
            bool(llm_verdict and assigned_label and llm_verdict != assigned_label)
        ),
        "disagree_llm_vs_model": int(
            bool(
                llm_verdict
                and model_pred_label
                and llm_verdict != model_pred_label
            )
        ),
    }


def export_reviewed_to_csv(db_path: Path, out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with sqlite3.connect(db_path) as conn, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        cols = [
            "source",
            "item_id",
            "split",
            "target",
            "assigned_label",
            "llm_verdict",
            "llm_confidence",
            "reason_code",
            "human_action",
            "final_label",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in conn.execute(
            """
            SELECT source,item_id,split,target,assigned_label,llm_verdict,llm_confidence,
                   reason_code,human_action,final_label
            FROM queue
            WHERE COALESCE(human_action, '') <> ''
            ORDER BY split,target,source,item_id
            """
        ):
            w.writerow(
                {
                    "source": row[0],
                    "item_id": row[1],
                    "split": row[2],
                    "target": row[3],
                    "assigned_label": row[4],
                    "llm_verdict": row[5],
                    "llm_confidence": row[6],
                    "reason_code": row[7],
                    "human_action": row[8],
                    "final_label": row[9],
                }
            )
            n += 1
    return n


def commit_queue_to_label_patches(
    *,
    db_path: Path,
    label_patches_path: Path,
    temp_csv_path: Path,
) -> dict[str, Any]:
    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            f"""
            SELECT source,item_id,target,assigned_label,llm_verdict,final_label,human_action
            FROM queue
            WHERE human_action IN ({human_action_sql_in_list(COMMIT_QUEUE_HUMAN_ACTIONS)})
            """
        )
        for (
            source,
            item_id,
            target,
            _assigned,
            llm_verdict,
            final_label,
            action,
        ) in cur:
            source = str(source)
            item_id = str(item_id)
            key = (source, item_id)
            row = rows_by_key.setdefault(
                key, {"source": source, "item_id": item_id}
            )
            new_label = (
                str(llm_verdict).strip()
                if action == "accept_llm"
                else str(final_label).strip()
            )
            if not new_label:
                continue
            row[f"{target}_label"] = new_label

    temp_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["item_id", "source", "sleeve_label", "media_label"]
        )
        w.writeheader()
        for r in rows_by_key.values():
            w.writerow(
                {
                    "item_id": r["item_id"],
                    "source": r["source"],
                    "sleeve_label": r.get("sleeve_label", ""),
                    "media_label": r.get("media_label", ""),
                }
            )
    patch_stats = append_csv_to_label_patches(temp_csv_path, label_patches_path)
    return {
        "grouped_rows": len(rows_by_key),
        "label_patches": patch_stats,
        "csv_path": str(temp_csv_path),
    }
