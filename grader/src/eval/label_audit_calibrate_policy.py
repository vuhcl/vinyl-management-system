from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grader.src.eval.label_audit_backend import ensure_db


SOURCE_NAMES = ("llm", "pred", "assigned")
TARGETS = ("sleeve", "media")
DEFAULT_REPORT_PATH = Path(
    "grader/reports/label_audit_auto_policy_report.json"
)
DEFAULT_DB = Path("grader/reports/label_audit_queue.sqlite")


def _utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _derive_ground_truth(row: sqlite3.Row) -> tuple[str, str]:
    action = str(row["human_action"] or "").strip()
    if action == "accept_llm":
        return str(row["llm_verdict"] or "").strip(), "accept_llm"
    if action == "keep_assigned":
        return str(row["assigned_label"] or "").strip(), "keep_assigned"
    if action in {"manual_set", "auto_apply"}:
        return str(row["final_label"] or "").strip(), action
    return "", ""


def _candidate_for_source(row: sqlite3.Row, source: str) -> tuple[str, float]:
    if source == "llm":
        label = str(row["llm_verdict"] or "").strip()
        abstain = int(row["llm_abstain"] or 0)
        if not label or abstain:
            return "", 0.0
        try:
            score = float(row["llm_confidence"] or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return label, max(0.0, min(1.0, score))
    if source == "pred":
        label = str(row["model_pred_label"] or "").strip()
        if not label:
            return "", 0.0
        try:
            score = float(row["model_pred_proba_max"] or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return label, max(0.0, min(1.0, score))
    label = str(row["assigned_label"] or "").strip()
    if not label:
        return "", 0.0
    try:
        score = float(row["model_proba_assigned"] or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return label, max(0.0, min(1.0, score))


def _parse_allowed_sources(raw: str) -> tuple[str, ...]:
    vals = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not vals:
        return SOURCE_NAMES
    allowed = tuple(x for x in vals if x in SOURCE_NAMES)
    if not allowed:
        raise ValueError(
            "allowed-sources must include at least one of: llm,pred,assigned"
        )
    return allowed


def _score_rows(
    rows: list[sqlite3.Row],
    thresholds: dict[str, dict[str, float]],
    allowed_sources: tuple[str, ...] = SOURCE_NAMES,
) -> dict[str, Any]:
    by_source: dict[str, dict[str, int]] = {
        s: {"proposed": 0, "correct": 0} for s in SOURCE_NAMES
    }
    proposed = 0
    correct = 0
    for row in rows:
        target = str(row["target"] or "").strip().lower()
        truth, _truth_source = _derive_ground_truth(row)
        if target not in TARGETS or not truth:
            continue
        candidates: list[tuple[str, str, float]] = []
        for source in allowed_sources:
            label, score = _candidate_for_source(row, source)
            if not label:
                continue
            min_score = float(thresholds[target][source])
            if target == "sleeve" and label == "Generic":
                min_score = max(
                    min_score, float(thresholds[target]["generic_min"])
                )
            if score >= min_score:
                candidates.append((source, label, score))
        if not candidates:
            continue
        source, label, score = max(candidates, key=lambda x: x[2])
        _ = score
        proposed += 1
        by_source[source]["proposed"] += 1
        if label == truth:
            correct += 1
            by_source[source]["correct"] += 1
    total = len(rows)
    precision = (correct / proposed) if proposed else 0.0
    coverage = (proposed / total) if total else 0.0
    for source in SOURCE_NAMES:
        p = by_source[source]["proposed"]
        c = by_source[source]["correct"]
        by_source[source]["precision"] = (c / p) if p else 0.0
    return {
        "precision": precision,
        "coverage": coverage,
        "proposed": proposed,
        "total": total,
        "correct": correct,
        "by_source": by_source,
    }


def _fit_thresholds(
    train_rows: list[sqlite3.Row],
    mode: str = "balanced",
    allowed_sources: tuple[str, ...] = SOURCE_NAMES,
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {
        "sleeve": {
            "llm": 0.7,
            "pred": 0.7,
            "assigned": 0.7,
            "generic_min": 0.95,
        },
        "media": {
            "llm": 0.7,
            "pred": 0.7,
            "assigned": 0.7,
            "generic_min": 1.0,
        },
    }
    if mode == "high_precision":
        thresholds["sleeve"]["llm"] = 0.8
        thresholds["sleeve"]["pred"] = 0.9
        thresholds["sleeve"]["assigned"] = 0.85
        thresholds["sleeve"]["generic_min"] = 0.97
        thresholds["media"]["llm"] = 0.8
        thresholds["media"]["pred"] = 0.9
        thresholds["media"]["assigned"] = 0.85
    grid = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    by_target: dict[str, list[sqlite3.Row]] = {"sleeve": [], "media": []}
    for row in train_rows:
        t = str(row["target"] or "").strip().lower()
        if t in by_target:
            by_target[t].append(row)
    for target in TARGETS:
        rows = by_target[target]
        if not rows:
            continue
        local = dict(thresholds[target])
        for source in SOURCE_NAMES:
            if source not in allowed_sources:
                # Disabled source: set threshold above max score to block use.
                local[source] = 1.01
                continue
            best_th = local[source]
            best_obj = float("-inf")
            for th in grid:
                local[source] = th
                other = {
                    k: thresholds[k] for k in TARGETS if k != target
                }
                stats = _score_rows(
                    rows,
                    {target: local, **other},
                    allowed_sources=allowed_sources,
                )
                precision = float(stats["precision"])
                coverage = float(stats["coverage"])
                if mode == "high_precision":
                    floor_penalty = -5.0 if precision < 0.9 else 0.0
                    objective = (
                        precision * 0.9
                        + coverage * 0.1
                        + floor_penalty
                    )
                else:
                    # Balanced objective with a hard-ish precision floor.
                    floor_penalty = -2.0 if precision < 0.9 else 0.0
                    objective = (
                        precision * 0.7
                        + coverage * 0.3
                        + floor_penalty
                    )
                if objective > best_obj:
                    best_obj = objective
                    best_th = th
            local[source] = best_th
        if target == "sleeve":
            best_generic = local["generic_min"]
            best_obj = float("-inf")
            for g in [0.9, 0.93, 0.95, 0.97, 0.99]:
                local["generic_min"] = g
                other = {
                    k: thresholds[k] for k in TARGETS if k != target
                }
                stats = _score_rows(
                    rows,
                    {target: local, **other},
                    allowed_sources=allowed_sources,
                )
                source_stats = stats["by_source"]
                pred_p = float(source_stats["pred"]["precision"])
                llm_p = float(source_stats["llm"]["precision"])
                objective = (
                    float(stats["precision"]) * 0.6
                    + float(stats["coverage"]) * 0.4
                )
                if max(pred_p, llm_p) < 0.95:
                    objective -= 0.5
                if mode == "high_precision" and float(stats["precision"]) < 0.9:
                    objective -= 1.0
                if objective > best_obj:
                    best_obj = objective
                    best_generic = g
            local["generic_min"] = best_generic
        if mode == "high_precision":
            if "llm" in allowed_sources:
                local["llm"] = max(local["llm"], 0.8)
            if "pred" in allowed_sources:
                local["pred"] = max(local["pred"], 0.9)
            if "assigned" in allowed_sources:
                local["assigned"] = max(local["assigned"], 0.85)
            if target == "sleeve":
                local["generic_min"] = max(local["generic_min"], 0.97)
        thresholds[target] = local
    return thresholds


def _stratified_split(
    rows: list[sqlite3.Row], holdout_ratio: float, seed: int
) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    rng = random.Random(seed)
    by_target: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        target = str(row["target"] or "").strip().lower()
        by_target[target].append(row)
    train: list[sqlite3.Row] = []
    holdout: list[sqlite3.Row] = []
    for target_rows in by_target.values():
        bucket = list(target_rows)
        rng.shuffle(bucket)
        if len(bucket) > 3:
            n_hold = max(1, int(round(len(bucket) * holdout_ratio)))
        else:
            n_hold = 1
        holdout.extend(bucket[:n_hold])
        train.extend(bucket[n_hold:])
    return train, holdout


def _load_reviewed_rows(
    conn: sqlite3.Connection, include_auto_apply: bool
) -> list[sqlite3.Row]:
    actions = ["accept_llm", "keep_assigned", "manual_set"]
    if include_auto_apply:
        actions.append("auto_apply")
    placeholders = ",".join("?" for _ in actions)
    rows = conn.execute(
        f"""
        SELECT queue_row_id,target,assigned_label,model_pred_label,
               model_pred_proba_max,model_proba_assigned,
               llm_verdict,llm_confidence,llm_abstain,reason_code,
               human_action,final_label,split,source,item_id
        FROM queue
        WHERE COALESCE(human_action, '') IN ({placeholders})
        """,
        tuple(actions),
    ).fetchall()
    return [r for r in rows if _derive_ground_truth(r)[0]]


def _apply_policy(
    conn: sqlite3.Connection,
    thresholds: dict[str, dict[str, float]],
    policy_version: str,
    dry_run: bool,
    allowed_sources: tuple[str, ...] = SOURCE_NAMES,
) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT queue_row_id,target,assigned_label,model_pred_label,
               model_pred_proba_max,
               model_proba_assigned,llm_verdict,llm_confidence,llm_abstain,reason_code,human_action
        FROM queue
        WHERE COALESCE(human_action, '') = ''
        """
    ).fetchall()
    proposed = 0
    review = 0
    by_source = {"pred": 0, "llm": 0, "assigned": 0}
    for row in rows:
        target = str(row["target"] or "").strip().lower()
        if target not in TARGETS:
            continue
        candidates: list[tuple[str, str, float]] = []
        for source in allowed_sources:
            label, score = _candidate_for_source(row, source)
            if not label:
                continue
            min_score = float(thresholds[target][source])
            if target == "sleeve" and label == "Generic":
                min_score = max(
                    min_score, float(thresholds[target]["generic_min"])
                )
            if score >= min_score:
                candidates.append((source, label, score))
        if candidates:
            src, label, score = max(candidates, key=lambda x: x[2])
            proposed += 1
            by_source[src] += 1
            reason = (
                f"source={src};score={score:.3f};threshold={thresholds[target][src]:.3f};"
                f"generic_min={thresholds[target]['generic_min']:.3f}"
            )[:500]
            if not dry_run:
                conn.execute(
                    """
                    UPDATE queue
                    SET auto_decision='proposed_grade',
                        auto_final_label=?,
                        auto_decision_score=?,
                        auto_decision_reason=?,
                        auto_decision_source=?,
                        auto_policy_version=?,
                        updated_at=datetime('now')
                    WHERE queue_row_id=? AND COALESCE(human_action, '') = ''
                    """,
                    (
                        label,
                        float(score),
                        reason,
                        src,
                        policy_version,
                        int(row["queue_row_id"]),
                    ),
                )
        else:
            review += 1
            if not dry_run:
                conn.execute(
                    """
                    UPDATE queue
                    SET auto_decision='needs_review',
                        auto_final_label='',
                        auto_decision_score=NULL,
                        auto_decision_reason='no candidate passed threshold',
                        auto_decision_source='',
                        auto_policy_version=?,
                        updated_at=datetime('now')
                    WHERE queue_row_id=? AND COALESCE(human_action, '') = ''
                    """,
                    (policy_version, int(row["queue_row_id"])),
                )
    if not dry_run:
        conn.commit()
    return {
        "proposed_grade": proposed,
        "needs_review": review,
        "proposed_from_pred": by_source["pred"],
        "proposed_from_llm": by_source["llm"],
        "proposed_from_assigned": by_source["assigned"],
    }


def _clear_policy_version(
    conn: sqlite3.Connection, policy_version: str, dry_run: bool
) -> int:
    rows = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM queue
        WHERE COALESCE(auto_policy_version, '') = ?
        """,
        (policy_version,),
    ).fetchone()
    count = int(rows["n"] if rows else 0)
    if dry_run:
        return count
    conn.execute(
        """
        UPDATE queue
        SET auto_decision='',
            auto_final_label='',
            auto_decision_score=NULL,
            auto_decision_reason='',
            auto_decision_source='',
            auto_policy_version='',
            updated_at=datetime('now')
        WHERE COALESCE(auto_policy_version, '') = ?
        """,
        (policy_version,),
    )
    conn.commit()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate/apply auto-review policy."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--policy-version", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-auto-apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clear-policy-version", default="")
    parser.add_argument(
        "--mode",
        choices=["balanced", "high_precision"],
        default="balanced",
    )
    parser.add_argument("--min-holdout-precision", type=float, default=-1.0)
    parser.add_argument("--allowed-sources", default="llm,pred,assigned")
    args = parser.parse_args()

    db_path = Path(args.db)
    ensure_db(db_path)
    policy_version = args.policy_version.strip() or f"auto-{_utc_compact()}"
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_sources = _parse_allowed_sources(args.allowed_sources)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if args.clear_policy_version.strip():
            n = _clear_policy_version(
                conn,
                policy_version=args.clear_policy_version.strip(),
                dry_run=bool(args.dry_run),
            )
            print(
                json.dumps(
                    {
                        "event": "clear_policy_version",
                        "policy_version": args.clear_policy_version.strip(),
                        "rows_cleared": n,
                        "dry_run": bool(args.dry_run),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        reviewed_rows = _load_reviewed_rows(
            conn, include_auto_apply=bool(args.include_auto_apply)
        )
        counts_by_target: dict[str, int] = defaultdict(int)
        for row in reviewed_rows:
            counts_by_target[str(row["target"] or "").strip().lower()] += 1
        if not args.force:
            if len(reviewed_rows) < 100:
                raise ValueError(
                    "Need at least 100 reviewed rows for calibration; "
                    f"got {len(reviewed_rows)}. Use --force to override."
                )
            for target in TARGETS:
                if counts_by_target[target] < 30:
                    raise ValueError(
                        "Need at least 30 reviewed rows for "
                        f"target={target}; got {counts_by_target[target]}. "
                        "Use --force to override."
                    )

        train_rows, holdout_rows = _stratified_split(
            reviewed_rows,
            holdout_ratio=float(args.holdout_ratio),
            seed=int(args.seed),
        )
        thresholds = _fit_thresholds(
            train_rows,
            mode=str(args.mode),
            allowed_sources=allowed_sources,
        )
        train_stats = _score_rows(
            train_rows, thresholds, allowed_sources=allowed_sources
        )
        holdout_stats = _score_rows(
            holdout_rows, thresholds, allowed_sources=allowed_sources
        )
        holdout_precision = float(holdout_stats.get("precision", 0.0))
        min_holdout_precision = float(args.min_holdout_precision)
        if min_holdout_precision < 0:
            min_holdout_precision = 0.9 if args.mode == "high_precision" else 0.0
        if args.apply and holdout_precision < min_holdout_precision:
            raise ValueError(
                "Refusing to apply policy: holdout precision "
                f"{holdout_precision:.3f} < required {min_holdout_precision:.3f}. "
                "Re-run with --dry-run, improve decisions, or use --force."
            )

        apply_counts = {
            "proposed_grade": 0,
            "needs_review": 0,
            "proposed_from_pred": 0,
            "proposed_from_llm": 0,
            "proposed_from_assigned": 0,
        }
        if args.apply:
            apply_counts = _apply_policy(
                conn,
                thresholds=thresholds,
                policy_version=policy_version,
                dry_run=bool(args.dry_run),
                allowed_sources=allowed_sources,
            )

    report = {
        "event": "calibration_report",
        "policy_version": policy_version,
        "db_path": str(db_path),
        "reviewed_rows": len(reviewed_rows),
        "reviewed_rows_by_target": dict(counts_by_target),
        "holdout_ratio": float(args.holdout_ratio),
        "seed": int(args.seed),
        "mode": str(args.mode),
        "allowed_sources": list(allowed_sources),
        "min_holdout_precision": float(min_holdout_precision),
        "thresholds": thresholds,
        "train": train_stats,
        "holdout": holdout_stats,
        "applied": bool(args.apply),
        "dry_run": bool(args.dry_run),
        "apply_counts": apply_counts,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
