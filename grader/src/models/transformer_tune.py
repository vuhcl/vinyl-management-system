"""
Hyperparameter sweeps for the DistilBERT grader (``transformer_tune``).

Presets live in ``grader/configs/transformer_tune.yaml``.

Option A: each preset trains with **local** MLflow (see ``mlflow.tuning_*`` in
``grader.yaml``) so the remote server is not flooded. After the sweep, the best
preset is registered on the **remote** tracking server by logging a new pyfunc
run from the on-disk artifacts under ``paths.artifacts/tuning/<preset>/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import yaml

from grader.src.mlflow_tracking import (
    configure_mlflow_from_config,
    mlflow_enabled,
    mlflow_log_artifacts_enabled,
    vinyl_grader_pyfunc_has_python_model,
)
from grader.src.models.grader_pyfunc import log_pyfunc_model
from grader.src.models.transformer import TransformerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_tuning_presets(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    presets = data.get("presets") or {}
    if not presets:
        raise ValueError(f"No presets found in {path}")
    return presets


def _merge_preset_config(
    base_cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    t = cfg.setdefault("models", {}).setdefault("transformer", {})
    for k, v in overrides.items():
        if k == "description":
            continue
        t[k] = v
    # Keep paths.artifacts at the pipeline root (label_encoder_*.pkl, features,
    # baseline pickles live there). Per-preset outputs go under tuning/<preset>/
    # via TransformerTrainer(..., artifact_subdir=...).
    return cfg


def _write_temp_config(cfg: dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="grader_tune_", suffix=".yaml")
    with open(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _eval_summary(eval_results: dict[str, Any]) -> dict[str, float]:
    ts = float(eval_results["test"]["sleeve"]["macro_f1"])
    tm = float(eval_results["test"]["media"]["macro_f1"])
    vs = float(eval_results["val"]["sleeve"]["macro_f1"])
    vm = float(eval_results["val"]["media"]["macro_f1"])
    trs = float(eval_results["train"]["sleeve"]["macro_f1"])
    trm = float(eval_results["train"]["media"]["macro_f1"])
    return {
        "test_sleeve_macro_f1": ts,
        "test_media_macro_f1": tm,
        "test_mean_macro_f1": (ts + tm) / 2.0,
        "val_mean_macro_f1": (vs + vm) / 2.0,
        "train_mean_macro_f1": (trs + trm) / 2.0,
        "sleeve_train_test_gap": trs - ts,
        "media_train_test_gap": trm - tm,
    }


def _append_results_csv(
    path: Path,
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def promote_preset(artifacts_dir: Path, preset_key: str) -> None:
    """Copy ``artifacts/tuning/<preset>/`` into the inference root."""
    src = artifacts_dir / "tuning" / preset_key
    if not src.is_dir():
        raise FileNotFoundError(
            f"Tune run not found: {src}. Train this preset first."
        )
    for item in src.iterdir():
        dest = artifacts_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    logger.info("Promoted %s → %s", src, artifacts_dir)


def _register_best_on_remote(
    base_config_path: str,
    preset_key: str,
    artifacts_dir: Path,
    registry_model_name: str,
) -> None:
    """Log pyfunc from disk under *remote* tracking, then register_model."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    art = artifacts_dir / "tuning" / preset_key
    w = art / "transformer_weights.pt"
    if not w.is_file():
        logger.warning("Skipping registry: no weights at %s", w)
        return
    configure_mlflow_from_config(cfg)
    fake = SimpleNamespace(
        weights_path=art / "transformer_weights.pt",
        config_path_out=art / "transformer_config.json",
        tokenizer_dir=art / "tokenizer",
        artifacts_dir=art,
    )
    with mlflow.start_run(run_name=f"tune_best_{preset_key}"):
        log_pyfunc_model(fake)
        run_id = mlflow.active_run().info.run_id
    if not vinyl_grader_pyfunc_has_python_model(run_id):
        logger.warning(
            "Skipping model registry: run %s has no usable pyfunc bundle.",
            run_id,
        )
        return
    model_uri = f"runs:/{run_id}/vinyl_grader"
    try:
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=registry_model_name,
            tags={
                "preset": preset_key,
                "source": "transformer_tune_remote_promote",
            },
        )
        logger.info(
            "Registered model '%s' version %s from %s",
            registry_model_name,
            mv.version,
            model_uri,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model registration failed: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for DistilBERT vinyl grader"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Base grader config",
    )
    parser.add_argument(
        "--tuning-config",
        default="grader/configs/transformer_tune.yaml",
        help="YAML with presets: mapping",
    )
    parser.add_argument(
        "--presets",
        default="",
        help="Comma-separated preset keys, or 'all'",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Do not create MLflow runs (same as --no-mlflow)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow entirely (same as mlflow.enabled: false).",
    )
    parser.add_argument(
        "--mlflow-no-artifacts",
        action="store_true",
        help=(
            "Params/metrics only — no pyfunc/GCS artifacts. "
            "Ignored if --no-mlflow or --skip-mlflow."
        ),
    )
    parser.add_argument(
        "--register",
        action="store_true",
        default=True,
        help=(
            "After sweep, log best preset to remote MLflow and register "
            "(default: true)"
        ),
    )
    parser.add_argument(
        "--no-register",
        dest="register",
        action="store_false",
        help="Skip remote model registry registration",
    )
    parser.add_argument(
        "--registry-model-name",
        default="VinylGrader",
        help="Registered model name in the MLflow model registry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="One epoch per preset, no saves (smoke test)",
    )
    parser.add_argument(
        "--results-csv",
        default="grader/reports/transformer_tune_results.csv",
        help="Append one row per completed preset",
    )
    parser.add_argument(
        "--promote",
        default="",
        help="Copy grader/artifacts/tuning/<name>/ to artifacts root and exit",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ml = cfg.setdefault("mlflow", {})
    if args.skip_mlflow or args.no_mlflow:
        ml["enabled"] = False
        logger.info("MLflow disabled (--skip-mlflow / --no-mlflow).")
    elif args.mlflow_no_artifacts and ml.get("enabled", True):
        ml["log_artifacts"] = False
        logger.info(
            "MLflow metrics-only (--mlflow-no-artifacts / "
            "mlflow.log_artifacts: false)."
        )

    artifacts_dir = Path(cfg["paths"]["artifacts"])
    Path("grader/experiments").mkdir(parents=True, exist_ok=True)

    if args.promote.strip():
        promote_preset(artifacts_dir, args.promote.strip())
        return

    all_presets = _load_tuning_presets(Path(args.tuning_config))
    if not args.presets.strip():
        parser.error(
            "Provide --presets (e.g. all or partial1_low_lr) or --promote"
        )

    if args.presets.strip().lower() == "all":
        chosen = list(all_presets.keys())
    else:
        chosen = [p.strip() for p in args.presets.split(",") if p.strip()]

    for key in chosen:
        if key not in all_presets:
            raise KeyError(
                f"Unknown preset {key!r}. Available: {sorted(all_presets)}"
            )

    fieldnames = [
        "timestamp_utc",
        "preset",
        "description",
        "test_sleeve_macro_f1",
        "test_media_macro_f1",
        "test_mean_macro_f1",
        "val_mean_macro_f1",
        "train_mean_macro_f1",
        "sleeve_train_test_gap",
        "media_train_test_gap",
        "best_val_mean_f1",
        "best_epoch",
        "hparams_json",
    ]

    run_registry: dict[str, dict[str, Any]] = {}
    skip_mlf = args.skip_mlflow or args.no_mlflow

    for key in chosen:
        spec = all_presets[key]
        desc = str(spec.get("description", ""))
        overrides = dict(spec)
        overrides.pop("description", None)

        logger.info("=== Tuning preset %s ===", key)
        logger.info("Overrides: %s", json.dumps(overrides, default=str))

        merged = _merge_preset_config(cfg, overrides)
        tmp_path = _write_temp_config(merged)
        try:
            trainer = TransformerTrainer(
                tmp_path,
                tuning=True,
                artifact_subdir=f"tuning/{key}",
            )
            out = trainer.run(
                dry_run=args.dry_run,
                skip_mlflow=skip_mlf,
                mlflow_run_name=f"tune_{key}",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        ev = out["eval"]
        tr = out["training"]
        summ = _eval_summary(ev)

        row = {
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "preset": key,
            "description": desc,
            **summ,
            "best_val_mean_f1": round(float(tr["best_val_f1"]), 6),
            "best_epoch": int(tr["best_epoch"]),
            "hparams_json": json.dumps(overrides, sort_keys=True, default=str),
        }
        _append_results_csv(Path(args.results_csv), fieldnames, row)
        logger.info(
            "Preset %s | test mean macro-F1: %.4f | sleeve gap: %.4f",
            key,
            summ["test_mean_macro_f1"],
            summ["sleeve_train_test_gap"],
        )

        if not skip_mlf and not args.dry_run:
            run_registry[key] = {
                "test_mean_macro_f1": summ["test_mean_macro_f1"],
                "preset": key,
            }

    if (
        args.register
        and mlflow_enabled(cfg)
        and not skip_mlf
        and not args.dry_run
        and mlflow_log_artifacts_enabled(cfg)
        and run_registry
    ):
        best = max(
            run_registry.values(),
            key=lambda x: x["test_mean_macro_f1"],
        )
        _register_best_on_remote(
            args.config,
            str(best["preset"]),
            artifacts_dir,
            args.registry_model_name,
        )


if __name__ == "__main__":
    main()
