"""List largest holdout prediction errors for a saved VinylIQ bundle.

Rebuilds the training frame using ``catalog_encoders.json`` from the model dir so feature
indices match the fit estimators, reproduces the same release-level split as
``train_vinyliq`` (tuning path when enabled), loads estimators, and prints the worst rows.

Usage (repo root)::

  PYTHONPATH=. uv run python price_estimator/scripts/worst_holdout_predictions.py
  PYTHONPATH=. uv run python price_estimator/scripts/worst_holdout_predictions.py \\
      --split val --top 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

from price_estimator.src.features.vinyliq_features import (
    default_feature_columns,
    residual_training_feature_columns,
)
from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    ensemble_blend_weight_log_anchor,
    load_fitted_regressor,
    log1p_dollar_targets_for_metrics,
    pred_log1p_dollar_for_metrics,
)
from price_estimator.src.training.train_vinyliq import (
    load_training_frame,
    residual_z_clip_abs_from_vinyliq,
    train_test_split_by_release,
    training_label_config_from_vinyliq,
    training_target_kind_from_vinyliq,
)


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml_config(path: Path | None) -> dict:
    if path is not None:
        p = path
    else:
        env = os.environ.get("VINYLIQ_CONFIG")
        p = Path(env) if env else _pkg_root() / "configs" / "base.yaml"
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(cfg: dict) -> tuple[Path, Path, Path, Path]:
    v = cfg.get("vinyliq") or {}
    paths = v.get("paths") or {}
    root = _pkg_root()
    d_cache = root / "data" / "cache"
    d_data = root / "data"
    mp = Path(
        paths.get("marketplace_db", d_cache / "marketplace_stats.sqlite"),
    )
    fs = Path(paths.get("feature_store_db", d_data / "feature_store.sqlite"))
    sh = Path(paths.get("sale_history_db", d_cache / "sale_history.sqlite"))
    md = Path(paths.get("model_dir", root / "artifacts" / "vinyliq"))
    if not mp.is_absolute():
        mp = root / mp
    if not fs.is_absolute():
        fs = root / fs
    if not sh.is_absolute():
        sh = root / sh
    if not md.is_absolute():
        md = root / md
    return mp, fs, sh, md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML config (default: VINYLIQ_CONFIG or "
            "price_estimator/configs/base.yaml)"
        ),
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override vinyliq.paths.model_dir",
    )
    ap.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Outer test holdout or inner tuning validation split",
    )
    ap.add_argument("--top", type=int, default=25, help="How many rows to print")
    ap.add_argument(
        "--by",
        choices=("abs", "ape"),
        default="abs",
        help="Rank by absolute dollar error or relative APE",
    )
    args = ap.parse_args()

    cfg = _load_yaml_config(args.config)
    v = cfg.get("vinyliq") or {}
    mp, fs, sh, md = _resolve_paths(cfg)
    if args.model_dir is not None:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = _pkg_root() / md

    ce_path = md / "catalog_encoders.json"
    if not ce_path.is_file():
        print(
            f"Missing {ce_path} (train once to emit encoders)",
            file=sys.stderr,
        )
        return 1
    enc_saved = json.loads(ce_path.read_text())

    manifest_path = md / "model_manifest.json"
    if not manifest_path.is_file():
        print(f"Missing {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    fc_path = md / "feature_columns.joblib"
    if not fc_path.is_file():
        print(f"Missing {fc_path}", file=sys.stderr)
        return 1
    feature_columns: list[str] = list(joblib.load(fc_path))

    training_label_cfg = training_label_config_from_vinyliq(v)
    target_kind = training_target_kind_from_vinyliq(v)
    z_clip = residual_z_clip_abs_from_vinyliq(v)
    ce_cfg = v.get("catalog_encoders") or {}
    max_art = ce_cfg.get("max_primary_artist_ids")
    max_lbl = ce_cfg.get("max_primary_label_ids")
    if max_art is not None:
        max_art = int(max_art)
    if max_lbl is not None:
        max_lbl = int(max_lbl)

    sh_arg = sh if sh.is_file() else None
    frame = load_training_frame(
        mp,
        fs,
        max_primary_artist_ids=max_art,
        max_primary_label_ids=max_lbl,
        training_label=training_label_cfg,
        training_target_kind=target_kind,
        residual_z_clip_abs=z_clip,
        sale_history_db=sh_arg,
        catalog_encoders_override=enc_saved,
    )

    cols_expect = (
        residual_training_feature_columns()
        if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN
        else default_feature_columns()
    )
    if list(feature_columns) != list(cols_expect):
        print(
            "Warning: feature_columns.joblib differs from config-derived columns "
            f"(artifact n={len(feature_columns)} vs cfg n={len(cols_expect)}). "
            "Using artifact columns.",
            file=sys.stderr,
        )

    cols = feature_columns
    X_all = np.array([[float(row[c]) for c in cols] for row in frame.xrows])
    y_all = np.asarray(frame.yvals, dtype=np.float64)
    median_all = np.asarray(frame.median_anchors, dtype=np.float64)
    rids = frame.rids

    tuning = v.get("tuning") or {}
    tuning_on = bool(tuning.get("enabled", False))
    if tuning_on:
        test_fraction = float(tuning.get("test_fraction", 0.15))
        val_fraction = float(tuning.get("val_fraction", 0.15))
        tune_seed = tuning.get("random_seed")
        seed = int(tune_seed) if tune_seed is not None else int(
            cfg.get("seed", 42)
        )
        train_r, test_r = train_test_split_by_release(
            rids, test_fraction=test_fraction, seed=seed
        )
        _inner_train_r, inner_val_r = train_test_split_by_release(
            list(train_r), test_fraction=val_fraction, seed=seed + 1
        )
    else:
        train_r, test_r = train_test_split_by_release(
            rids, test_fraction=0.15, seed=int(cfg.get("seed", 42))
        )
        inner_val_r = set()

    if args.split == "test":
        mask = np.array([rid in test_r for rid in rids])
        split_desc = "outer test"
    else:
        if not tuning_on:
            print(
                "--split val requires vinyliq.tuning.enabled",
                file=sys.stderr,
            )
            return 1
        mask = np.array([rid in inner_val_r and rid in train_r for rid in rids])
        split_desc = "inner val"

    X_sub = X_all[mask]
    y_sub = y_all[mask]
    med_sub = median_all[mask]
    rids_sub = [r for r, m in zip(rids, mask, strict=True) if m]

    schema = int(manifest.get("schema_version", 1))
    ens_raw = manifest.get("ensemble")
    ens = ens_raw if isinstance(ens_raw, dict) else {}
    is_ensemble = schema >= 3 and bool(ens.get("enabled"))

    if is_ensemble:
        blend = ens.get("blend") if isinstance(ens.get("blend"), dict) else {}
        t_b = float(blend.get("t", 0.0))
        s_b = float(blend.get("s", 1.0))
        nm_rel = str(ens.get("regressor_nm", "regressor_ensemble_nm.joblib"))
        ord_rel = str(ens.get("regressor_ord", "regressor_ensemble_ord.joblib"))
        est_nm = joblib.load(md / nm_rel)
        est_ord = joblib.load(md / ord_rel)
        pred_nm = np.asarray(est_nm.predict(X_sub), dtype=np.float64).ravel()
        pred_ord = np.asarray(est_ord.predict(X_sub), dtype=np.float64).ravel()
        pred_lp_nm = pred_log1p_dollar_for_metrics(
            pred_nm, med_sub, target_kind
        )
        pred_lp_ord = pred_log1p_dollar_for_metrics(
            pred_ord, med_sub, target_kind
        )
        w = ensemble_blend_weight_log_anchor(
            med_sub, center_log1p=t_b, scale=s_b
        )
        pred_lp = w * pred_lp_nm + (1.0 - w) * pred_lp_ord
    else:
        fitted = load_fitted_regressor(md)
        if fitted is None:
            print(
                "Could not load fitted regressor from model dir",
                file=sys.stderr,
            )
            return 1
        pred_raw = fitted.predict_log1p(X_sub)
        pred_lp = pred_log1p_dollar_for_metrics(pred_raw, med_sub, target_kind)

    y_lp = log1p_dollar_targets_for_metrics(y_sub, med_sub, target_kind)
    y_d = np.expm1(np.maximum(y_lp, 0.0))
    p_d = np.expm1(np.maximum(pred_lp, 0.0))
    abs_err = np.abs(p_d - y_d)
    floor = 1.0
    ape = abs_err / np.maximum(y_d, floor)

    if args.by == "abs":
        order = np.argsort(-abs_err)
    else:
        order = np.argsort(-ape)

    print(
        f"Worst predictions ({split_desc}, n={int(mask.sum())}, "
        f"ensemble={is_ensemble}, target_kind={target_kind}, by={args.by})"
    )
    hdr = (
        "release_id\ty_true_usd\ty_pred_usd\tabs_err_usd\t"
        "ape_pct\tmedian_anchor_usd"
    )
    print(hdr)
    top_n = min(args.top, len(order))
    for i in range(top_n):
        j = int(order[i])
        rid = rids_sub[j]
        print(
            f"{rid}\t{y_d[j]:.2f}\t{p_d[j]:.2f}\t{abs_err[j]:.2f}\t"
            f"{100.0 * ape[j]:.2f}\t{med_sub[j]:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
