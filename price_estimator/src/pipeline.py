"""
VinylIQ entrypoints: training wrapper and local estimate() for backwards compatibility.

Train:
  PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq

Seed demo DBs:
  python price_estimator/scripts/seed_demo_data.py

API server:
  PYTHONPATH=. uvicorn price_estimator.src.api.main:app --host 0.0.0.0 --port 8801
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict:
    p = config_path or _root() / "configs" / "base.yaml"
    with open(p) as f:
        return yaml.safe_load(f) or {}


def run_pipeline(
    config_path: Path | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run VinylIQ training (single XGB or multi-model tuning per ``vinyliq.tuning.enabled``)."""
    from .training.train_vinyliq import main as train_main

    if config_path:
        import os

        os.environ["VINYLIQ_CONFIG"] = str(config_path)
    code = train_main()
    if code != 0:
        raise RuntimeError(f"train_vinyliq exited with {code}")
    cfg = load_config(config_path)
    v = cfg.get("vinyliq") or {}
    paths = v.get("paths") or {}
    root = _root()
    md = Path(paths.get("model_dir", root / "artifacts" / "vinyliq"))
    if not md.is_absolute():
        md = root / md
    return {"model_path": str(md), "status": "trained"}


def estimate(
    release_id: str,
    sleeve_condition: str | None = None,
    media_condition: str | None = None,
    artifacts_dir: Path | None = None,
    **kwargs: Any,
) -> dict:
    """
    Local estimate using InferenceService (same logic as POST /estimate).
    Ignores artifacts_dir (VinylIQ uses vinyliq.model_dir from config).
    """
    from .inference.service import InferenceService

    cfg = load_config()
    v = cfg.get("vinyliq") or {}
    paths = v.get("paths") or {}
    root = _root()
    mp = Path(paths.get("marketplace_db", root / "data" / "cache" / "marketplace_stats.sqlite"))
    fs = Path(paths.get("feature_store_db", root / "data" / "feature_store.sqlite"))
    md = Path(paths.get("model_dir", root / "artifacts" / "vinyliq"))
    if not mp.is_absolute():
        mp = root / mp
    if not fs.is_absolute():
        fs = root / fs
    if not md.is_absolute():
        md = root / md
    svc = InferenceService(
        marketplace_db=mp,
        feature_store_db=fs,
        model_dir=md,
    )
    out = svc.estimate(str(release_id), media_condition, sleeve_condition)
    return {
        "release_id": out.get("release_id", release_id),
        "sleeve_condition": sleeve_condition,
        "media_condition": media_condition,
        "estimate_usd": out.get("estimated_price"),
        "interval_low": out.get("confidence_interval", [None, None])[0],
        "interval_high": out.get("confidence_interval", [None, None])[1]
        if out.get("confidence_interval")
        else None,
        "baseline_median": out.get("baseline_median"),
        "model_version": out.get("model_version"),
        "status": out.get("status", "ok"),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="VinylIQ price_estimator")
    p.add_argument("command", choices=["train"], nargs="?", default="train")
    args = p.parse_args()
    if args.command == "train":
        print(run_pipeline())
