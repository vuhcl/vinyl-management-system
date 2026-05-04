"""Rule engine evaluation step (extracted from pipeline)."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np

from grader.src.evaluation.grade_analysis import (
    build_grade_analysis_report,
    build_rule_owned_slice_report,
    resolve_rule_owned_grades,
    slice_recall_for_grade,
)
from grader.src.evaluation.metrics import (
    compute_metrics_from_label_strings,
    compute_rule_override_audit,
    format_override_audit_report,
    remap_true_and_encode_predictions,
    substitute_model_when_pred_excellent,
)
from grader.src.features.tfidf_features import TFIDFFeatureBuilder
from grader.src.models.baseline import BaselineModel
from grader.src.models.transformer import TransformerTrainer
from grader.src.rules.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


def current_git_sha() -> Optional[str]:
    """Return short git HEAD sha, or None if git is unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return sha.decode("utf-8", errors="replace").strip() or None
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def write_rule_engine_baseline(path: Path, snapshot: dict) -> None:
    """Persist the rule-engine baseline snapshot as canonical JSON."""
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "commit": current_git_sha(),
        "splits": snapshot,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def tag_rule_engine_baseline(snapshot: dict) -> None:
    """Mirror key baseline numbers to MLflow tags (stringified)."""
    if not mlflow.active_run():
        return
    for split_name, targets in snapshot.items():
        for target, data in targets.items():
            base = f"rule_baseline_{split_name}_{target}"
            prec = data.get("override_precision")
            if prec is not None:
                mlflow.set_tag(f"{base}_override_precision", f"{prec:.4f}")
            mlflow.set_tag(
                f"{base}_rule_owned_grades",
                ",".join(data.get("rule_owned_grades", [])),
            )
            for g, row in (data.get("by_after") or {}).items():
                gsafe = g.lower().replace(" ", "_")
                mlflow.set_tag(
                    f"{base}_harmful_to_{gsafe}",
                    str(row.get("n_harmful", 0)),
                )
                rp = row.get("override_precision")
                if rp is not None:
                    mlflow.set_tag(
                        f"{base}_override_precision_to_{gsafe}",
                        f"{rp:.4f}",
                    )
            for g, vals in (data.get("slice_recall") or {}).items():
                gsafe = g.lower().replace(" ", "_")
                ra = vals.get("recall_adjusted")
                if ra is not None:
                    mlflow.set_tag(
                        f"{base}_true_{gsafe}_recall_adjusted",
                        f"{ra:.4f}",
                    )


def run_rule_engine_evaluation(
    pipeline: Any,
    rule_engine: RuleEngine,
    trainer: Optional[TransformerTrainer],
    baseline: BaselineModel,
    use_transformer: bool,
) -> tuple[dict, dict[str, float], dict[str, str]]:
    """
    Rule-adjusted metrics, model-only metrics, and override audit on
    configured evaluation splits.
    """
    features_dir = str(pipeline.artifacts_dir / "features")
    splits_dir = Path(pipeline.config["paths"]["splits"])
    out: dict = {}
    mlflow_flat: dict[str, float] = {}
    grade_analysis_paths: dict[str, str] = {}
    use_excellent_blend = bool(
        pipeline.config.get("evaluation", {}).get(
            "excellent_eval_use_model_prediction", False
        )
    )
    rule_owned_grades = resolve_rule_owned_grades(rule_engine.guidelines)
    # Baseline snapshot accumulator: {split: {target: {...}}}
    baseline_snapshot: dict[str, dict] = {}

    ev = pipeline.config.get("evaluation") or {}
    _res = ev.get("rule_eval_splits")
    if isinstance(_res, list) and _res:
        eval_splits = [str(s) for s in _res]
    else:
        eval_splits = ["test", "test_thin"]

    for split_name in eval_splits:
        if split_name == "test_thin" and not (splits_dir / "test_thin.jsonl").exists():
            logger.info(
                "Rule eval — skip split=test_thin (no test_thin.jsonl)"
            )
            continue
        if split_name == "test_thin":
            try:
                TFIDFFeatureBuilder.load_features(
                    features_dir, split="test_thin", target="sleeve"
                )
            except OSError:
                logger.info(
                    "Rule eval — skip split=test_thin (no feature matrices)"
                )
                continue

        raw, texts = pipeline._predictions_for_rule_eval(
            split_name, trainer, baseline, use_transformer
        )
        adjusted = rule_engine.apply_batch(raw, texts)
        coverage = rule_engine.summarize_results(adjusted)

        adjusted_m: dict[str, dict] = {}
        adjusted_raw_m: dict[str, dict] = {}
        model_m: dict[str, dict] = {}
        audit_m: dict[str, dict] = {}
        grade_report_sections: list[str] = []

        for target in ("sleeve", "media"):
            _, y = TFIDFFeatureBuilder.load_features(
                features_dir, split=split_name, target=target
            )
            encoder = TFIDFFeatureBuilder.load_encoder(
                str(pipeline.artifacts_dir / f"label_encoder_{target}.pkl")
            )
            pred_key = f"predicted_{target}_condition"
            before = [str(p[pred_key]) for p in raw]
            after = [str(p[pred_key]) for p in adjusted]
            after_eval = (
                substitute_model_when_pred_excellent(after, before)
                if use_excellent_blend
                else after
            )
            n_ex_subst = (
                sum(1 for a, e in zip(after, after_eval) if a != e)
                if use_excellent_blend
                else 0
            )
            adjusted_m[target] = compute_metrics_from_label_strings(
                y,
                after_eval,
                encoder.classes_,
                target=target,
                split=split_name,
            )
            if use_excellent_blend:
                adjusted_raw_m[target] = compute_metrics_from_label_strings(
                    y,
                    after,
                    encoder.classes_,
                    target=target,
                    split=split_name,
                )
            model_m[target] = compute_metrics_from_label_strings(
                y,
                before,
                encoder.classes_,
                target=target,
                split=split_name,
            )
            audit_m[target] = compute_rule_override_audit(
                y,
                before,
                after,
                encoder.classes_,
                target=target,
                split=split_name,
            )
            logger.info(
                "Rule eval — split=%s target=%s | model macro-F1 %.4f → "
                "adjusted %.4f (Δ %+.4f) | helpful=%d harmful=%d neutral=%d "
                "override_precision=%s | excellent→model rows=%d",
                split_name,
                target,
                model_m[target]["macro_f1"],
                adjusted_m[target]["macro_f1"],
                audit_m[target]["delta_macro_f1"],
                audit_m[target]["n_helpful"],
                audit_m[target]["n_harmful"],
                audit_m[target]["n_neutral"],
                audit_m[target]["override_precision"],
                n_ex_subst,
            )
            grade_report_sections.append(
                build_grade_analysis_report(
                    y,
                    before,
                    after,
                    encoder.classes_,
                    target=target,
                    split=split_name,
                    after_for_scoring=after_eval
                    if use_excellent_blend
                    else None,
                )
            )

            # Rule-owned slice section (true-label-conditioned view)
            owned_for_target = rule_owned_grades.get(target, [])
            grade_report_sections.append(
                build_rule_owned_slice_report(
                    y,
                    before,
                    after,
                    encoder.classes_,
                    target=target,
                    split=split_name,
                    rule_owned_grades=owned_for_target,
                )
            )

            # Formatted override-audit section with by_after /
            # by_transition breakdowns (compact text tables).
            grade_report_sections.append(
                format_override_audit_report(audit_m[target])
            )

            # Compute slice recalls for rule-owned grades — used
            # both for MLflow tags/metrics and for the baseline JSON.
            y_t2, combined_list, (y_b_idx, y_a_idx) = (
                remap_true_and_encode_predictions(
                    y, encoder.classes_, before, after
                )
            )
            combined_arr = np.array(combined_list)
            slice_recalls: dict[str, dict] = {}
            for g in owned_for_target:
                slice_recalls[g] = {
                    "recall_model": slice_recall_for_grade(
                        y_t2, y_b_idx, combined_arr, g
                    ),
                    "recall_adjusted": slice_recall_for_grade(
                        y_t2, y_a_idx, combined_arr, g
                    ),
                }
            audit_m[target]["slice_recall"] = slice_recalls
            baseline_snapshot.setdefault(split_name, {})[target] = {
                "rule_owned_grades": owned_for_target,
                "override_precision": audit_m[target][
                    "override_precision"
                ],
                "n_changed": audit_m[target]["n_changed"],
                "n_helpful": audit_m[target]["n_helpful"],
                "n_harmful": audit_m[target]["n_harmful"],
                "n_neutral": audit_m[target]["n_neutral"],
                "delta_macro_f1": audit_m[target].get("delta_macro_f1"),
                "delta_accuracy": audit_m[target].get("delta_accuracy"),
                "by_after": audit_m[target].get("by_after", {}),
                "by_transition": audit_m[target].get(
                    "by_transition", {}
                ),
                "slice_recall": slice_recalls,
            }

        reports_dir = Path(pipeline.config["paths"]["reports"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        gap = "\n\n" + "=" * 72 + "\n\n"
        ga_path = reports_dir / f"grade_analysis_{split_name}.txt"
        ga_path.write_text(
            gap.join(grade_report_sections),
            encoding="utf-8",
        )
        grade_analysis_paths[split_name] = str(ga_path)
        logger.info("Grade analysis report — %s", ga_path)

        out[split_name] = {
            "adjusted": adjusted_m,
            "model":    model_m,
            "audit":    audit_m,
            "coverage": coverage,
        }
        if use_excellent_blend:
            out[split_name]["adjusted_raw"] = adjusted_raw_m

        sk = split_name
        for target in ("sleeve", "media"):
            adj = adjusted_m[target]
            aud = audit_m[target]
            mlflow_flat[f"rule_adjusted_{sk}_{target}_macro_f1"] = float(
                adj["macro_f1"]
            )
            mlflow_flat[f"rule_adjusted_{sk}_{target}_accuracy"] = float(
                adj["accuracy"]
            )
            if use_excellent_blend:
                rawm = adjusted_raw_m[target]
                mlflow_flat[
                    f"rule_adjusted_raw_{sk}_{target}_macro_f1"
                ] = float(rawm["macro_f1"])
                mlflow_flat[
                    f"rule_adjusted_raw_{sk}_{target}_accuracy"
                ] = float(rawm["accuracy"])
            mlflow_flat[f"rule_model_{sk}_{target}_macro_f1"] = float(
                model_m[target]["macro_f1"]
            )
            mlflow_flat[f"rule_model_{sk}_{target}_accuracy"] = float(
                model_m[target]["accuracy"]
            )
            mlflow_flat[f"rule_audit_{sk}_{target}_delta_macro_f1"] = float(
                aud["delta_macro_f1"]
            )
            mlflow_flat[f"rule_audit_{sk}_{target}_delta_accuracy"] = float(
                aud["delta_accuracy"]
            )
            mlflow_flat[f"rule_audit_{sk}_{target}_helpful"] = float(
                aud["n_helpful"]
            )
            mlflow_flat[f"rule_audit_{sk}_{target}_harmful"] = float(
                aud["n_harmful"]
            )
            mlflow_flat[f"rule_audit_{sk}_{target}_neutral"] = float(
                aud["n_neutral"]
            )
            if aud["override_precision"] is not None:
                mlflow_flat[
                    f"rule_audit_{sk}_{target}_override_precision"
                ] = float(aud["override_precision"])

            # Rule-owned slice metrics (§6).
            for g, vals in (aud.get("slice_recall") or {}).items():
                gsafe = g.lower().replace(" ", "_")
                rm = vals.get("recall_model")
                ra = vals.get("recall_adjusted")
                if rm is not None:
                    mlflow_flat[
                        f"rule_slice_{sk}_{target}_true_{gsafe}_recall_model"
                    ] = float(rm)
                if ra is not None:
                    mlflow_flat[
                        f"rule_slice_{sk}_{target}_true_{gsafe}_recall_adjusted"
                    ] = float(ra)

            # Stratified "harmful to <grade>" metrics, capped to
            # the top-K destinations that actually saw overrides.
            by_after = aud.get("by_after") or {}
            for g, row in by_after.items():
                gsafe = g.lower().replace(" ", "_")
                mlflow_flat[
                    f"rule_audit_{sk}_{target}_harmful_to_{gsafe}"
                ] = float(row.get("n_harmful", 0))
                prec = row.get("override_precision")
                if prec is not None:
                    mlflow_flat[
                        f"rule_audit_{sk}_{target}_override_precision_to_{gsafe}"
                    ] = float(prec)

    if "test" in out:
        for k in ("sleeve", "media"):
            mlflow_flat[f"rule_adjusted_{k}_macro_f1"] = float(
                out["test"]["adjusted"][k]["macro_f1"]
            )
            mlflow_flat[f"rule_adjusted_{k}_accuracy"] = float(
                out["test"]["adjusted"][k]["accuracy"]
            )

    # --- §8 Baseline snapshot: JSON artifact + MLflow tags ----------
    if baseline_snapshot:
        reports_dir = Path(pipeline.config["paths"]["reports"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        snap_path = reports_dir / "rule_engine_baseline.json"
        write_rule_engine_baseline(
            snap_path, baseline_snapshot
        )
        logger.info("Rule engine baseline snapshot — %s", snap_path)
        try:
            tag_rule_engine_baseline(baseline_snapshot)
        except Exception as exc:  # mlflow-optional path
            logger.debug(
                "Skipped MLflow baseline tags (no active run?): %s", exc
            )

    return out, mlflow_flat, grade_analysis_paths
