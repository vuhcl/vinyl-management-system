"""CLI and non-tuning training path for VinylIQ."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ...features.vinyliq_features import (
    default_feature_columns,
    residual_training_feature_columns,
)
from ...mlflow_tracking import configure_mlflow_from_config
from ...models.fitted_regressor import (
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    fit_regressor,
    log1p_dollar_targets_for_metrics,
    mae_dollars,
    median_ape_dollars,
    pred_log1p_dollar_for_metrics,
    wape_dollars,
)
from ...models.xgb_vinyliq import XGBVinylIQModel
from ..label_synthesis import training_label_config_from_vinyliq
from .catalog_encoders import _write_encoder_artifacts
from ..confidence_artifacts import write_confidence_training_bundle
from .training_config import (
    _config_path_for_mlflow,
    _mlflow_flags,
    _training_label_console_summary,
    _training_label_mlflow_params,
    ensemble_blend_config_from_vinyliq,
    load_config,
    residual_z_clip_abs_from_vinyliq,
    training_target_kind_from_vinyliq,
    _root,
    _write_training_label_config,
)
from .training_frame import load_training_frame, report_residual_target_sanity
from .release_train_split import train_test_split_by_release
from .tuning_runner import _run_tuning


def main(args: argparse.Namespace | None = None) -> int:
    cfg = load_config()
    if args is not None:
        if getattr(args, "no_mlflow", False):
            cfg.setdefault("mlflow", {})["enabled"] = False
        elif getattr(args, "mlflow_no_artifacts", False):
            cfg.setdefault("mlflow", {})["log_artifacts"] = False
    v = cfg.get("vinyliq") or {}
    paths = v.get("paths") or {}
    root = _root()
    mp = Path(paths.get("marketplace_db", root / "data" / "cache" / "marketplace_stats.sqlite"))
    fs = Path(paths.get("feature_store_db", root / "data" / "feature_store.sqlite"))
    sh = Path(paths.get("sale_history_db", root / "data" / "cache" / "sale_history.sqlite"))
    md = Path(paths.get("model_dir", root / "artifacts" / "vinyliq"))
    if not mp.is_absolute():
        mp = root / mp
    if not fs.is_absolute():
        fs = root / fs
    if not sh.is_absolute():
        sh = root / sh
    if not md.is_absolute():
        md = root / md
    md.mkdir(parents=True, exist_ok=True)

    if not mp.exists() or not fs.exists():
        print("Need marketplace_stats.sqlite and feature_store.sqlite first.", file=sys.stderr)
        return 1

    ce_cfg = v.get("catalog_encoders") or {}
    max_art = ce_cfg.get("max_primary_artist_ids")
    max_lbl = ce_cfg.get("max_primary_label_ids")
    if max_art is not None:
        max_art = int(max_art)
    if max_lbl is not None:
        max_lbl = int(max_lbl)

    training_label_cfg = training_label_config_from_vinyliq(v)
    if args is not None:
        ovr = getattr(args, "sale_condition_policy", None)
        if ovr:
            pol = str(ovr).strip().lower()
            if pol in ("nm_substrings_only", "ordinal_cascade"):
                sfb = training_label_cfg.setdefault("sale_floor_blend", {})
                if not isinstance(sfb, dict):
                    training_label_cfg["sale_floor_blend"] = {}
                    sfb = training_label_cfg["sale_floor_blend"]
                sfb["sale_condition_policy"] = pol
                print(
                    f"CLI override: sale_floor_blend.sale_condition_policy={pol!r}",
                    file=sys.stderr,
                )
    target_kind = training_target_kind_from_vinyliq(v)
    raw_tt = v.get("training_target")
    training_target_artifact: dict[str, object] = (
        {**raw_tt, "kind": target_kind}
        if isinstance(raw_tt, dict)
        else {"kind": target_kind}
    )
    print(f"Training label: {_training_label_console_summary(training_label_cfg)}")
    _mode_l = str(training_label_cfg.get("mode", "")).strip().lower()
    if _mode_l in ("sale_floor_blend", "sale_floor"):
        print(
            f"Sale history DB (for sold nowcast): {sh} "
            f"(exists={sh.is_file()}; rows need fetch_status ok + NM-filtered sales)"
        )
    print(f"Training target kind: {target_kind}")
    z_clip = residual_z_clip_abs_from_vinyliq(v)
    if z_clip is not None:
        print(f"Residual z clip (abs): {z_clip}")

    sh_arg = sh if sh.is_file() else None
    ensemble_cfg = ensemble_blend_config_from_vinyliq(v)
    frame = load_training_frame(
        mp,
        fs,
        max_primary_artist_ids=max_art,
        max_primary_label_ids=max_lbl,
        training_label=training_label_cfg,
        training_target_kind=target_kind,
        residual_z_clip_abs=z_clip,
        sale_history_db=sh_arg,
    )
    if ensemble_cfg is not None:
        y_nm_ct = int(
            np.sum(np.isfinite(np.asarray(frame.yvals_nm, dtype=np.float64)))
        )
        y_ord_ct = int(
            np.sum(np.isfinite(np.asarray(frame.yvals_ord, dtype=np.float64)))
        )
        if y_nm_ct < 20 or y_ord_ct < 20:
            print(
                "Ensemble needs >=20 rows per policy with a valid sale-floor label "
                f"(NM-substrings: {y_nm_ct}, ordinal-cascade: {y_ord_ct}; "
                f"primary-labeled n={len(frame.rids)}).",
                file=sys.stderr,
            )
            return 1
        print(
            "Ensemble: NM head trains on NM-policy labels only, ordinal head on "
            f"ordinal-policy labels (rows with label: NM={y_nm_ct}, Ord={y_ord_ct}; "
            f"primary-labeled n={len(frame.rids)}) — aligned with serving both heads on every row.",
        )

    Xrows = frame.xrows
    yvals = frame.yvals
    rids = frame.rids
    catalog_encoders = frame.catalog_encoders
    median_anchors = frame.median_anchors
    has_nm_comp_arr = np.asarray(frame.has_nm_comp_sale, dtype=np.float64)
    has_ord_comp_arr = np.asarray(frame.has_ord_comp_sale, dtype=np.float64)
    y_nm_arr = np.asarray(frame.yvals_nm, dtype=np.float64)
    y_ord_arr = np.asarray(frame.yvals_ord, dtype=np.float64)
    meta = catalog_encoders.get("_id_encoder_meta") or {}
    if meta:
        print(
            "Catalog id encoders: "
            f"n_labeled={int(meta.get('n_labeled_rows', 0))}, "
            f"artist unique={int(meta.get('primary_artist_id_unique', 0))}, "
            f"artist top_k={int(meta.get('primary_artist_id_cap', 0))}, "
            f"label unique={int(meta.get('primary_label_id_unique', 0))}, "
            f"label top_k={int(meta.get('primary_label_id_cap', 0))}",
        )
    if len(Xrows) < 20:
        print(f"Not enough labeled rows with features: {len(Xrows)} (need >= 20)", file=sys.stderr)
        return 1

    cols = (
        residual_training_feature_columns()
        if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN
        else default_feature_columns()
    )
    X_all = np.array([[float(row[c]) for c in cols] for row in Xrows])
    y_all = np.array(yvals)
    median_all = np.asarray(median_anchors, dtype=np.float64)
    report_residual_target_sanity(
        y_all,
        median_all,
        target_kind,
        seed=int(cfg.get("seed", 42)),
    )
    if args is not None and getattr(args, "residual_sanity_only", False):
        if target_kind != TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            print(
                "--residual-sanity-only requires vinyliq.training_target.kind: "
                "residual_log_median",
                file=sys.stderr,
            )
            return 1
        print(
            "Exiting after residual sanity (--residual-sanity-only).",
            file=sys.stderr,
        )
        return 0

    tuning = v.get("tuning") or {}
    if ensemble_cfg is not None and not tuning.get("enabled", False):
        print(
            "vinyliq.ensemble.enabled requires vinyliq.tuning.enabled=true "
            "(ensemble uses champion hyperparameters from the tuning loop).",
            file=sys.stderr,
        )
        return 1
    if tuning.get("enabled", False):
        return _run_tuning(
            cfg,
            root,
            md,
            X_all,
            y_all,
            median_all,
            rids,
            catalog_encoders,
            cols,
            training_label_cfg,
            target_kind=target_kind,
            has_nm_comp_sale=has_nm_comp_arr,
            has_ord_comp_sale=has_ord_comp_arr,
            y_nm=y_nm_arr,
            y_ord=y_ord_arr,
            ensemble_cfg=ensemble_cfg,
        )

    train_r, test_r = train_test_split_by_release(
        rids, test_fraction=0.15, seed=cfg.get("seed", 42)
    )
    train_mask = np.array([rid in train_r for rid in rids])
    test_mask = np.array([rid in test_r for rid in rids])

    seed = int(cfg.get("seed", 42))
    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        xgb_cfg = v.get("xgboost") or {}
        xgb_params = xgb_cfg if isinstance(xgb_cfg, dict) else {}
        fitted, _ = fit_regressor(
            "xgboost",
            xgb_params,
            X_all[train_mask],
            y_all[train_mask],
            cols,
            random_state=seed,
            target_kind=target_kind,
        )
        fitted.save(md)
        pred = fitted.predict_log1p(X_all[test_mask])
        y_te = y_all[test_mask]
        med_te = median_all[test_mask]
        y_te_lp = log1p_dollar_targets_for_metrics(y_te, med_te, target_kind)
        pred_lp = pred_log1p_dollar_for_metrics(pred, med_te, target_kind)
        mae = mae_dollars(y_te_lp, pred_lp)
        wape = wape_dollars(y_te_lp, pred_lp)
        mdape = median_ape_dollars(y_te_lp, pred_lp)
    else:
        model = XGBVinylIQModel()
        xgb_cfg = v.get("xgboost") or {}
        model.fit(
            X_all[train_mask],
            y_all[train_mask],
            feature_columns=cols,
            xgb_params=xgb_cfg if isinstance(xgb_cfg, dict) else None,
        )
        model.save(md)

        pred = model.predict_log1p(X_all[test_mask])
        y_te = y_all[test_mask]
        mae = mae_dollars(y_te, pred)
        wape = wape_dollars(y_te, pred)
        mdape = median_ape_dollars(y_te, pred)
    print(
        f"Holdout MAE $ {mae:.4f} | WAPE {100.0 * wape:.2f}% | median APE {100.0 * mdape:.2f}%"
    )

    if target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        write_confidence_training_bundle(
            model_dir=md,
            vinyliq_cfg=v,
            target_kind=target_kind,
            champion=fitted,
            cols=cols,
            seed=seed,
            x_train=X_all[train_mask],
            y_train=y_all[train_mask],
            sample_weight=None,
            y_test=y_te,
            pred_test=pred,
            median_test=median_all[test_mask],
            ensemble_active=ensemble_cfg is not None,
            log1p_y_test=None,
            log1p_pred_test=None,
        )
    else:
        write_confidence_training_bundle(
            model_dir=md,
            vinyliq_cfg=v,
            target_kind=target_kind,
            champion=None,
            cols=cols,
            seed=seed,
            x_train=X_all[train_mask],
            y_train=y_all[train_mask],
            sample_weight=None,
            y_test=y_te,
            pred_test=pred,
            median_test=None,
            ensemble_active=ensemble_cfg is not None,
            log1p_y_test=None,
            log1p_pred_test=None,
        )

    _write_encoder_artifacts(md, catalog_encoders)
    _write_training_label_config(
        md,
        training_label_cfg,
        training_target=training_target_artifact,
    )

    cfg_path = _config_path_for_mlflow(root)
    mflow_on, mflow_art = _mlflow_flags(cfg)

    if not mflow_on:
        print("MLflow disabled; model saved locally only.", file=sys.stderr)
    else:
        try:
            import mlflow

            configure_mlflow_from_config(cfg)
            mlflow_cfg = cfg.get("mlflow") or {}
            tags = mlflow_cfg.get("tags") or {}
            with mlflow.start_run(run_name="vinyliq_xgb_train"):
                if tags:
                    mlflow.set_tags({str(k): str(v) for k, v in tags.items()})
                if not mflow_art:
                    mlflow.log_param("mlflow_log_artifacts", "false")
                mlflow.log_metric("test_mae_dollars_approx", mae)
                mlflow.log_metric("test_wape_dollars", wape)
                mlflow.log_metric("test_median_ape_dollars", mdape)
                params: dict[str, str] = {
                    "model": "xgboost_vinyliq",
                    "n_train": str(int(train_mask.sum())),
                    "n_test": str(int(test_mask.sum())),
                    "training_target_kind": str(target_kind),
                }
                params.update(_training_label_mlflow_params(training_label_cfg))
                xgb_cfg = v.get("xgboost") or {}
                if isinstance(xgb_cfg, dict):
                    for k, val in xgb_cfg.items():
                        params[f"xgb_{k}"] = str(val)
                mlflow.log_params(params)
                if mflow_art:
                    if cfg_path.is_file():
                        mlflow.log_artifact(str(cfg_path), artifact_path="config")
                    mlflow.log_artifacts(str(md), artifact_path="vinyliq_model")
                else:
                    print(
                        "MLflow metrics-only: skipped artifact upload for xgb train.",
                        file=sys.stderr,
                    )
        except Exception as e:
            print(f"MLflow logging skipped: {e}", file=sys.stderr)

    print(f"Saved model to {md}")
    return 0


def _cli_main() -> int:
    parser = argparse.ArgumentParser(description="Train VinylIQ price model.")
    parser.add_argument(
        "--google-application-credentials",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "GCP service account JSON for MLflow artifact upload to GCS "
            "(sets GOOGLE_APPLICATION_CREDENTIALS for this process)."
        ),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow entirely (same as mlflow.enabled: false in config).",
    )
    parser.add_argument(
        "--mlflow-no-artifacts",
        action="store_true",
        help=(
            "Log params/metrics only; do not upload model bundle to artifact store "
            "(same as mlflow.log_artifacts: false). Ignored if --no-mlflow."
        ),
    )
    parser.add_argument(
        "--residual-sanity-only",
        action="store_true",
        help=(
            "Load training frame, print residual z stats and constant-z/shuffled-anchor "
            "baselines, then exit (no tuning or model fit)."
        ),
    )
    parser.add_argument(
        "--sale-condition-policy",
        choices=("nm_substrings_only", "ordinal_cascade"),
        default=None,
        help=(
            "Override vinyliq.training_label.sale_floor_blend.sale_condition_policy for "
            "MLflow A/B (same YAML file; logged as training_label_sf_* params)."
        ),
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}", file=sys.stderr)

    if args.google_application_credentials is not None:
        cred = args.google_application_credentials.expanduser().resolve()
        if not cred.is_file():
            print(
                f"--google-application-credentials: not a file: {cred}",
                file=sys.stderr,
            )
            return 1
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred)

    return main(args)
