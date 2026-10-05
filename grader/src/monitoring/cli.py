"""CLI: score a model on a split, gate challenger vs champion, check cohort drift.

Run from repo root:

  # 1. score each model's artifacts on the held-out split and the thin cohort
  python -m grader.src.monitoring.cli score --split test      --artifact-subdir ""                 --out grader/reports/monitoring/champion_test.jsonl
  python -m grader.src.monitoring.cli score --split test      --artifact-subdir retrain/challenger --out grader/reports/monitoring/challenger_test.jsonl
  python -m grader.src.monitoring.cli score --split test_thin --artifact-subdir ""                 --out grader/reports/monitoring/champion_test_thin.jsonl

  # 2. gate (exit code 0 = promote, 3 = keep champion)
  python -m grader.src.monitoring.cli gate \
      --champion grader/reports/monitoring/champion_test.jsonl \
      --challenger grader/reports/monitoring/challenger_test.jsonl \
      --report grader/reports/monitoring/gate_report.json

  # 3. cohort drift: test vs test_thin, scored by the champion
  python -m grader.src.monitoring.cli drift \
      --reference grader/reports/monitoring/champion_test.jsonl \
      --current   grader/reports/monitoring/champion_test_thin.jsonl \
      --report grader/reports/monitoring/drift_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

DEFAULT_THRESHOLDS = Path(__file__).resolve().parent / "thresholds.yaml"


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _thresholds(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cmd_score(args: argparse.Namespace) -> int:
    from grader.src.config_io import load_yaml_mapping
    from grader.src.models.transformer import TransformerTrainer

    cfg = load_yaml_mapping(args.config)
    split_path = Path(cfg["paths"]["splits"]) / f"{args.split}.jsonl"
    records = _load_jsonl(str(split_path))

    cfg.setdefault("mlflow", {})["enabled"] = False  # scoring never needs a tracking server
    subdir = args.artifact_subdir.strip() or None
    trainer = TransformerTrainer(
        config_path=args.config, artifact_subdir=subdir, config=cfg
    )
    trainer.encoders = trainer.load_encoders()
    trainer.load_model()

    texts = [r.get("text_clean") or r.get("text") or "" for r in records]
    ids = [str(r.get("item_id", i)) for i, r in enumerate(records)]

    if args.rules:
        # Full inference path: preprocess -> transformer -> rule engine (what is served).
        # Requires a reachable MLflow URI for Pipeline init; point it at a local store, e.g.
        # MLFLOW_TRACKING_URI=sqlite:///grader/experiments/mlflow_local.db
        from grader.src.pipeline.pipeline import Pipeline

        pipe = Pipeline(config_path=args.config)
        pipe._transformer = trainer  # use the chosen artifacts, not the default dir
        raw = [r.get("text") or r.get("text_clean") or "" for r in records]
        metas = [
            {"sleeve_label": r["sleeve_label"], "media_label": r["media_label"]}
            for r in records
        ]
        preds = {
            p["item_id"]: p
            for p in pipe.predict_batch(texts=raw, item_ids=ids, metadata_list=metas)
        }
    else:
        preds = {
            p["item_id"]: p
            for p in trainer.predict(texts=texts, item_ids=ids, records=records)
        }

    conf = lambda p, t: max(p["confidence_scores"][t].values())

    out = []
    for r, iid, text in zip(records, ids, texts):
        p = preds[iid]
        out.append(
            {
                "item_id": iid,
                "true_sleeve": r["sleeve_label"],
                "true_media": r["media_label"],
                "pred_sleeve": p["predicted_sleeve_condition"],
                "pred_media": p["predicted_media_condition"],
                "sleeve_conf": conf(p, "sleeve"),
                "media_conf": conf(p, "media"),
                "text_len": len(text),
            }
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows to {args.out}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from grader.src.monitoring.retrain_gate import evaluate_gate, macro_f1

    gate_cfg = _thresholds(args.thresholds)["retrain_gate"]
    report = evaluate_gate(_load_jsonl(args.champion), _load_jsonl(args.challenger), gate_cfg)

    # Non-gating: report both models on the thin cohort when files are given.
    if args.champion_thin and args.challenger_thin:
        c, h = _load_jsonl(args.champion_thin), _load_jsonl(args.challenger_thin)
        report["test_thin_report_only"] = {
            t: {"champion": round(macro_f1(c, t), 4), "challenger": round(macro_f1(h, t), 4)}
            for t in ("sleeve", "media")
        }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if report["promote"] and args.promote_model:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        version = client.get_latest_versions(args.promote_model)[-1].version if not args.promote_version else args.promote_version
        client.set_registered_model_alias(args.promote_model, args.alias, str(version))
        print(f"alias '{args.alias}' -> {args.promote_model} v{version}")
    return 0 if report["promote"] else 3


def cmd_drift(args: argparse.Namespace) -> int:
    from grader.src.monitoring.cohort_drift import cohort_drift

    cfg = _thresholds(args.thresholds)["cohort_drift"]
    report = cohort_drift(_load_jsonl(args.reference), _load_jsonl(args.current), cfg)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--config", default="grader/configs/grader.yaml")
    s.add_argument("--split", required=True, choices=("val", "test", "test_thin"))
    s.add_argument("--artifact-subdir", default="")
    s.add_argument("--out", required=True)
    s.add_argument("--rules", action="store_true", help="score through the rule engine (served behaviour)")
    s.set_defaults(fn=cmd_score)

    g = sub.add_parser("gate")
    g.add_argument("--champion", required=True)
    g.add_argument("--challenger", required=True)
    g.add_argument("--champion-thin", default="")
    g.add_argument("--challenger-thin", default="")
    g.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    g.add_argument("--report", required=True)
    g.add_argument("--promote-model", default="", help="MLflow registered model name; set alias on promote")
    g.add_argument("--promote-version", default="")
    g.add_argument("--alias", default="production")
    g.set_defaults(fn=cmd_gate)

    d = sub.add_parser("drift")
    d.add_argument("--reference", required=True)
    d.add_argument("--current", required=True)
    d.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    d.add_argument("--report", required=True)
    d.set_defaults(fn=cmd_drift)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
