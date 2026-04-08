"""
Train VinylIQ regressor on marketplace labels + feature store.

Usage (from repo root):
  PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq
  PYTHONPATH=. python -m price_estimator.src.training.train_vinyliq \\
    --google-application-credentials /path/to/service-account.json

For remote MLflow with GCS artifacts, set ``GOOGLE_APPLICATION_CREDENTIALS`` in ``.env``,
``mlflow.google_application_credentials`` in YAML (repo-relative path), or the CLI flag above.

Set ``vinyliq.tuning.enabled: true`` in config for multi-model search + MLflow registry.

MLflow cost control: ``mlflow.enabled: false`` or ``--no-mlflow`` skips tracking entirely.
``mlflow.log_artifacts: false`` or ``--mlflow-no-artifacts`` logs params/metrics only (no model
bundle / pyfunc / registry uploads to GCS).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml

from ..features.vinyliq_features import (
    default_feature_columns,
    first_artist_id,
    first_label_id,
    residual_training_feature_columns,
    row_dict_for_inference,
)
from ..mlflow_tracking import configure_mlflow_from_config
from ..models.condition_adjustment import default_params, save_params
from ..models.fitted_regressor import (
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    fit_regressor,
    log1p_dollar_targets_for_metrics,
    mae_dollars,
    median_ape_dollars,
    median_ape_dollar_quartiles,
    median_ape_train_median_baseline,
    pred_log1p_dollar_for_metrics,
    refit_champion,
    wape_dollars,
)
from ..models.vinyliq_pyfunc import (
    VinylIQPricePyFunc,
    build_pyfunc_input_example,
    pyfunc_artifacts_dict,
)
from ..models.xgb_vinyliq import XGBVinylIQModel
from .label_synthesis import synthesize_training_price, training_label_config_from_vinyliq
from .search_space import sample_from_space


def training_target_kind_from_vinyliq(v: dict | None) -> str:
    raw = (v or {}).get("training_target") or {}
    if not isinstance(raw, dict):
        raw = {}
    k = str(raw.get("kind", "residual_log_median")).strip().lower()
    if k in ("residual_log_median", "residual", "residual_log1p_median"):
        return TARGET_KIND_RESIDUAL_LOG_MEDIAN
    return TARGET_KIND_DOLLAR_LOG1P


def residual_z_clip_abs_from_vinyliq(v: dict | None) -> float | None:
    """Optional winsor on residual ``z = log1p(y) - log1p(median)`` to ``[-c, c]`` (null = off)."""
    raw = (v or {}).get("training_target") or {}
    if not isinstance(raw, dict):
        return None
    c = raw.get("residual_z_clip_abs")
    if c is None:
        return None
    try:
        x = float(c)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config() -> dict:
    env = os.environ.get("VINYLIQ_CONFIG")
    p = Path(env) if env else (_root() / "configs" / "base.yaml")
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _mlflow_flags(cfg: dict) -> tuple[bool, bool]:
    """
    Returns ``(tracking_enabled, upload_artifacts)``.

    ``upload_artifacts`` is False when tracking is off. When true, config dir + model dir
    and pyfunc are uploaded; registry requires this path.
    """
    ml = cfg.get("mlflow") or {}
    on = bool(ml.get("enabled", True))
    art = bool(ml.get("log_artifacts", True))
    return on, art and on


def _config_path_for_mlflow(root: Path) -> Path:
    env_cfg = os.environ.get("VINYLIQ_CONFIG")
    if env_cfg:
        cfg_path = Path(env_cfg)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        return cfg_path
    return root / "configs" / "base.yaml"


def _auto_top_k_id_encoder(n_labeled: int, n_unique: int) -> int:
    if n_unique <= 0:
        return 0
    if n_unique <= 500:
        return n_unique
    k = max(500, min(3000, n_labeled // 25))
    return min(k, n_unique)


def _fit_frequency_capped_id_encoder(ids: list[str], max_k: int) -> dict[str, float]:
    if max_k <= 0:
        return {}
    c = Counter(i for i in ids if i)
    if not c:
        return {}
    top = [pid for pid, _ in c.most_common(max_k)]
    return {pid: float(i + 1) for i, pid in enumerate(top)}


def load_training_frame(
    marketplace_db: Path,
    feature_store_db: Path,
    *,
    max_primary_artist_ids: int | None = None,
    max_primary_label_ids: int | None = None,
    training_label: dict[str, object] | None = None,
    training_target_kind: str = TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    residual_z_clip_abs: float | None = None,
) -> tuple[
    list[dict],
    list[float],
    list[str],
    dict[str, dict[str, float]],
    list[float],
]:
    tl = training_label or {"mode": "median", "blend_median_weight": 0.7}
    mode = str(tl.get("mode", "median"))
    blend_w = float(tl.get("blend_median_weight", 0.7))
    spread_floor = tl.get("spread_lowest_floor_ratio")
    spread_min_mw = tl.get("spread_min_median_weight")
    spread_n_ref = tl.get("spread_num_for_sale_reference")

    conn_m = sqlite3.connect(str(marketplace_db))
    conn_m.row_factory = sqlite3.Row
    cur = conn_m.execute(
        """
        SELECT release_id, median_price, lowest_price, num_for_sale
        FROM marketplace_stats
        WHERE median_price IS NOT NULL AND median_price > 0
        """
    )
    labels: dict[str, float] = {}
    medians: dict[str, float] = {}
    for r in cur.fetchall():
        rid = str(r["release_id"])
        y = synthesize_training_price(
            r["median_price"],
            r["lowest_price"],
            mode=mode,
            blend_median_weight=blend_w,
            spread_lowest_floor_ratio=spread_floor,
            spread_min_median_weight=spread_min_mw,
            num_for_sale=r["num_for_sale"],
            spread_num_for_sale_reference=spread_n_ref,
        )
        if y is not None and y > 0:
            labels[rid] = float(y)
            medians[rid] = float(r["median_price"])
    conn_m.close()

    conn_f = sqlite3.connect(str(feature_store_db))
    conn_f.row_factory = sqlite3.Row
    cur = conn_f.execute("SELECT * FROM releases_features")
    rows = [dict(r) for r in cur.fetchall()]
    conn_f.close()

    labeled: list[dict] = []
    for r in rows:
        rid = str(r.get("release_id", ""))
        if rid not in labels:
            continue
        y = labels[rid]
        if y <= 0:
            continue
        labeled.append(r)

    artist_ids_per_row = [first_artist_id(r) for r in labeled]
    label_ids_per_row = [first_label_id(r) for r in labeled]
    n_art_u = len({x for x in artist_ids_per_row if x})
    n_lbl_u = len({x for x in label_ids_per_row if x})
    n_lab = len(labeled)
    if max_primary_artist_ids is not None:
        ka = max(0, int(max_primary_artist_ids))
    else:
        ka = _auto_top_k_id_encoder(n_lab, n_art_u)
    if max_primary_label_ids is not None:
        kl = max(0, int(max_primary_label_ids))
    else:
        kl = _auto_top_k_id_encoder(n_lab, n_lbl_u)

    genres_set: set[str] = set()
    countries_set: set[str] = set()
    for r in labeled:
        g = str(r.get("genre") or "").strip().lower()
        if g:
            genres_set.add(g)
        c = str(r.get("country") or "").strip().lower()
        if c:
            countries_set.add(c)

    catalog_encoders: dict[str, dict[str, float]] = {
        "genre": {g: float(i) for i, g in enumerate(sorted(genres_set))},
        "country": {c: float(i) for i, c in enumerate(sorted(countries_set))},
        "primary_artist_id": _fit_frequency_capped_id_encoder(artist_ids_per_row, ka),
        "primary_label_id": _fit_frequency_capped_id_encoder(label_ids_per_row, kl),
    }
    catalog_encoders["_id_encoder_meta"] = {
        "n_labeled_rows": float(n_lab),
        "primary_artist_id_unique": float(n_art_u),
        "primary_label_id_unique": float(n_lbl_u),
        "primary_artist_id_cap": float(ka),
        "primary_label_id_cap": float(kl),
    }
    g2i = catalog_encoders["genre"]
    c2i = catalog_encoders["country"]
    a2i = catalog_encoders["primary_artist_id"]
    l2i = catalog_encoders["primary_label_id"]

    inc_mkt = training_target_kind == TARGET_KIND_DOLLAR_LOG1P
    Xrows: list[dict] = []
    yvals: list[float] = []
    rids: list[str] = []
    median_anchors: list[float] = []
    for r in labeled:
        rid = str(r.get("release_id", ""))
        y_dollar = labels[rid]
        mp = medians[rid]
        gidx = g2i.get(str(r.get("genre") or "").strip().lower(), 0.0)
        cidx = c2i.get(str(r.get("country") or "").strip().lower(), 0.0)
        aidx = a2i.get(first_artist_id(r), 0.0)
        lbidx = l2i.get(first_label_id(r), 0.0)
        stats = {"median_price": 0.0, "lowest_price": 0.0, "num_for_sale": 0}
        row = row_dict_for_inference(
            rid,
            "Near Mint (NM or M-)",
            "Near Mint (NM or M-)",
            stats,
            r,
            genre_index=gidx,
            country_index=cidx,
            primary_artist_index=aidx,
            primary_label_index=lbidx,
            include_marketplace_scalars_in_features=inc_mkt,
        )
        Xrows.append(row)
        if training_target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
            z = float(np.log1p(y_dollar) - np.log1p(mp))
            if residual_z_clip_abs is not None and residual_z_clip_abs > 0:
                z = float(np.clip(z, -residual_z_clip_abs, residual_z_clip_abs))
            yvals.append(z)
        else:
            yvals.append(float(np.log1p(y_dollar)))
        median_anchors.append(float(mp))
        rids.append(rid)

    return Xrows, yvals, rids, catalog_encoders, median_anchors


def report_residual_target_sanity(
    y_stored: np.ndarray,
    median_anchors: np.ndarray,
    target_kind: str,
    *,
    seed: int = 42,
) -> None:
    """
    Print residual ``z`` distribution, dollar metrics for constant ``z=0`` (correct anchor),
    and dollar metrics when the median anchor is shuffled (pred still ``z=0``).

    If ``|z|`` is almost always ~0, tuned models can match ~0 MdAPE without leakage.
    Shuffled-anchor MdAPE should be large unless all medians are similar or ``z`` cancels.
    """
    if target_kind != TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        return
    y = np.asarray(y_stored, dtype=np.float64)
    m = np.asarray(median_anchors, dtype=np.float64)
    az = np.abs(y)
    n = len(y)
    print("\n--- Residual target sanity ---")
    print(f"n={n}")
    print(
        f"z: mean={float(np.mean(y)):.6f}  std={float(np.std(y)):.6f}  "
        f"median|z|={float(np.median(az)):.6f}"
    )
    qs = (5, 25, 50, 75, 95, 99)
    pct = np.percentile(az, qs)
    qstr = " | ".join(f"P{p}:{float(v):.5f}" for p, v in zip(qs, pct))
    print(f"|z| percentiles: {qstr}")
    frac_tiny = float(np.mean(az < 1e-6))
    frac_small = float(np.mean(az < 0.01))
    print(
        f"frac |z|<1e-6: {100.0 * frac_tiny:.2f}%  "
        f"frac |z|<0.01: {100.0 * frac_small:.2f}%"
    )
    log1pm = np.log1p(np.maximum(m, 0.0))
    print(
        f"log1p(median_price): std={float(np.std(log1pm)):.5f}  "
        f"min..max={float(np.min(log1pm)):.3f}..{float(np.max(log1pm)):.3f}"
    )
    pred_z0 = np.zeros_like(y)
    y_lp = log1p_dollar_targets_for_metrics(y, m, target_kind)
    pred_lp_ok = pred_log1p_dollar_for_metrics(pred_z0, m, target_kind)
    mae0 = mae_dollars(y_lp, pred_lp_ok)
    md0 = median_ape_dollars(y_lp, pred_lp_ok)
    w0 = wape_dollars(y_lp, pred_lp_ok)
    print(
        "Constant z=0 + correct per-row anchor → "
        f"MAE ${mae0:.4f} | WAPE {100.0 * w0:.2f}% | median APE {100.0 * md0:.2f}%"
    )
    rng = np.random.default_rng(int(seed))
    m_bad = rng.permutation(m)
    pred_lp_bad = pred_log1p_dollar_for_metrics(pred_z0, m_bad, target_kind)
    mae_bad = mae_dollars(y_lp, pred_lp_bad)
    md_bad = median_ape_dollars(y_lp, pred_lp_bad)
    print(
        "Constant z=0 + shuffled anchor (destructive check) → "
        f"MAE ${mae_bad:.4f} | median APE {100.0 * md_bad:.2f}% "
        "(expect >> 0 if anchors and labels align per row)"
    )
    print("--- (end residual sanity) ---\n")


def train_test_split_by_release(
    rids: list[str],
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(rids))
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_fraction))
    test_set = set(uniq[:n_test])
    train_set = set(uniq[n_test:])
    return train_set, test_set


def _write_encoder_artifacts(model_dir: Path, catalog_encoders: dict[str, dict[str, float]]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "catalog_encoders.json").write_text(json.dumps(catalog_encoders, indent=2))
    (model_dir / "genre_encoder.json").write_text(
        json.dumps(catalog_encoders.get("genre", {}), indent=2),
    )
    save_params(model_dir / "condition_params.json", default_params())


def _write_training_label_config(
    model_dir: Path,
    training_label: dict[str, object],
    *,
    training_target: dict[str, object] | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"schema_version": 1, **training_label}
    if training_target:
        payload["training_target"] = dict(training_target)
    (model_dir / "training_label.json").write_text(json.dumps(payload, indent=2))


def _resolve_tuning_selection_metric(
    tuning: dict | None,
) -> tuple[str, str]:
    """
    How to pick the champion trial (all metrics minimized).

    Returns ``(score_field, mlflow_name)`` where ``score_field`` is one of
    ``mae``, ``wape``, ``mdape`` mapping to validation metrics on the trial.
    """
    raw = str((tuning or {}).get("selection_metric", "median_ape")).strip().lower()
    aliases: dict[str, tuple[str, str]] = {
        "median_ape": ("mdape", "val_median_ape_dollars"),
        "mdape": ("mdape", "val_median_ape_dollars"),
        "val_median_ape_dollars": ("mdape", "val_median_ape_dollars"),
        "wape": ("wape", "val_wape_dollars"),
        "val_wape_dollars": ("wape", "val_wape_dollars"),
        "mae_dollars": ("mae", "val_mae_dollars_approx"),
        "mae": ("mae", "val_mae_dollars_approx"),
        "val_mae_dollars_approx": ("mae", "val_mae_dollars_approx"),
    }
    return aliases.get(raw, ("mdape", "val_median_ape_dollars"))


def _enabled_families(v: dict) -> list[str]:
    mf = v.get("model_families") or {}
    order = [
        "xgboost",
        "lightgbm",
        "catboost",
        "sklearn_hist_gbrt",
        "sklearn_rf",
        "sklearn_et",
    ]
    out: list[str] = []
    for name in order:
        if mf.get(name, False):
            out.append(name)
    return out


def _run_tuning(
    cfg: dict,
    root: Path,
    md: Path,
    X_all: np.ndarray,
    y_all: np.ndarray,
    median_all: np.ndarray,
    rids: list[str],
    catalog_encoders: dict[str, dict[str, float]],
    cols: list[str],
    training_label_cfg: dict[str, object],
    *,
    target_kind: str,
) -> int:
    v = cfg.get("vinyliq") or {}
    tuning = v.get("tuning") or {}
    test_fraction = float(tuning.get("test_fraction", 0.15))
    val_fraction = float(tuning.get("val_fraction", 0.15))
    n_trials = int(tuning.get("n_trials_per_family", 8))
    es = tuning.get("early_stopping_rounds")
    es_int = int(es) if es is not None else None
    tune_seed = tuning.get("random_seed")
    seed = int(tune_seed) if tune_seed is not None else int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    train_r, test_r = train_test_split_by_release(
        rids, test_fraction=test_fraction, seed=seed
    )
    inner_train_r, inner_val_r = train_test_split_by_release(
        list(train_r), test_fraction=val_fraction, seed=seed + 1
    )

    train_mask = np.array([rid in train_r for rid in rids])
    test_mask = np.array([rid in test_r for rid in rids])
    tune_train_mask = np.array([rid in inner_train_r for rid in rids])
    val_mask = np.array([rid in inner_val_r for rid in rids])

    X_tr_full = X_all[train_mask]
    y_tr_full = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    X_tt = X_all[tune_train_mask & train_mask]
    y_tt = y_all[tune_train_mask & train_mask]
    X_v = X_all[val_mask & train_mask]
    y_v = y_all[val_mask & train_mask]
    med = np.asarray(median_all, dtype=np.float64)
    med_tt = med[tune_train_mask & train_mask]
    med_v = med[val_mask & train_mask]
    med_test = med[test_mask]
    med_tr_full = med[train_mask]

    spaces = v.get("search_spaces") or {}
    families = _enabled_families(v)
    sel_field, sel_mlflow = _resolve_tuning_selection_metric(tuning)

    best: dict[str, object] = {
        "selection_score": float("inf"),
        "val_mae": float("nan"),
        "val_wape": float("nan"),
        "val_mdape": float("nan"),
        "family": "",
        "params": {},
        "best_iteration": None,
    }

    mlflow_cfg = cfg.get("mlflow") or {}
    mflow_on, mflow_art = _mlflow_flags(cfg)
    cfg_path = _config_path_for_mlflow(root)

    if mflow_on:
        import mlflow
        from mlflow.tracking import MlflowClient

        configure_mlflow_from_config(cfg)
        parent_ctx = mlflow.start_run(
            run_name="vinyliq_train", tags={"orchestration": "parent"}
        )
    else:
        mlflow = None  # type: ignore[assignment, misc]
        MlflowClient = None  # type: ignore[assignment, misc]
        parent_ctx = nullcontext()
        print("MLflow disabled (mlflow.enabled: false); training without remote tracking.")

    with parent_ctx:
        tags = mlflow_cfg.get("tags") or {}
        if mflow_on:
            if tags:
                mlflow.set_tags({str(k): str(v) for k, v in tags.items()})
            mlflow.set_tag("orchestration", "parent")
            mlflow.log_param("training_label_mode", str(training_label_cfg.get("mode", "median")))
            mlflow.log_param(
                "training_label_blend_median_weight",
                str(training_label_cfg.get("blend_median_weight", 0.7)),
            )
            mlflow.log_param(
                "training_label_spread_lowest_floor_ratio",
                str(training_label_cfg.get("spread_lowest_floor_ratio")),
            )
            mlflow.log_param(
                "training_label_spread_min_median_weight",
                str(training_label_cfg.get("spread_min_median_weight")),
            )
            mlflow.log_param(
                "training_label_spread_num_for_sale_reference",
                str(training_label_cfg.get("spread_num_for_sale_reference")),
            )
            mlflow.log_param("n_train_outer", int(train_mask.sum()))
            mlflow.log_param("n_test_outer", int(test_mask.sum()))
            mlflow.log_param("n_tune_train", int((tune_train_mask & train_mask).sum()))
            mlflow.log_param("n_tune_val", int((val_mask & train_mask).sum()))
            mlflow.log_param("tuning_selection_metric", sel_mlflow)
            if not mflow_art:
                mlflow.log_param("mlflow_log_artifacts", "false")
            mlflow.log_param("training_target_kind", str(target_kind))
            zc = residual_z_clip_abs_from_vinyliq(v)
            if zc is not None:
                mlflow.log_param("residual_z_clip_abs", str(zc))

        y_tt_lp = log1p_dollar_targets_for_metrics(y_tt, med_tt, target_kind)
        y_v_lp = log1p_dollar_targets_for_metrics(y_v, med_v, target_kind)
        val_mdape_train_median_bl = median_ape_train_median_baseline(y_tt_lp, y_v_lp)
        if mflow_on:
            mlflow.log_metric(
                "val_median_ape_train_median_log_baseline", val_mdape_train_median_bl
            )
        print(
            "Baseline (predict train median log1p on all val rows): "
            f"val median APE {100.0 * val_mdape_train_median_bl:.2f}% "
            "(if model is near this, learn a signal before chasing hparams)"
        )
        if mflow_on and not mflow_art:
            print(
                "MLflow metrics-only (mlflow.log_artifacts: false): "
                "skipping model bundle / pyfunc / registry uploads."
            )

        trial_run = 0
        for family in families:
            space = spaces.get(family)
            if not isinstance(space, dict) or not space:
                continue
            for _ in range(n_trials):
                params = sample_from_space(space, rng)
                trial_run += 1
                run_name = f"trial_{family}_{trial_run}"
                trial_ctx = (
                    mlflow.start_run(nested=True, run_name=run_name)
                    if mflow_on
                    else nullcontext()
                )
                with trial_ctx:
                    if mflow_on:
                        mlflow.log_param("model_family", family)
                        for pk, pv in params.items():
                            mlflow.log_param(f"hparam_{pk}", str(pv))
                    try:
                        reg, meta = fit_regressor(
                            family,
                            params,
                            X_tt,
                            y_tt,
                            cols,
                            X_val=X_v,
                            y_val=y_v,
                            early_stopping_rounds=es_int,
                            random_state=seed,
                            target_kind=target_kind,
                        )
                        pred_v = reg.predict_log1p(X_v)
                        # Reconstruct log1p(dollar) before $ metrics (residual: pred_z+log1p(median)).
                        y_v_lp_m = log1p_dollar_targets_for_metrics(y_v, med_v, target_kind)
                        pred_v_lp = pred_log1p_dollar_for_metrics(pred_v, med_v, target_kind)
                        val_mae = mae_dollars(y_v_lp_m, pred_v_lp)
                        val_wape = wape_dollars(y_v_lp_m, pred_v_lp)
                        val_mdape = median_ape_dollars(y_v_lp_m, pred_v_lp)
                        if mflow_on:
                            mlflow.log_metric("val_mae_dollars_approx", val_mae)
                            mlflow.log_metric("val_wape_dollars", val_wape)
                            mlflow.log_metric("val_median_ape_dollars", val_mdape)
                        bi = meta.get("best_iteration")
                        if mflow_on and bi is not None:
                            mlflow.log_metric("best_iteration", float(bi))
                        scores = {"mae": val_mae, "wape": val_wape, "mdape": val_mdape}
                        trial_score = float(scores[sel_field])
                        if math.isnan(trial_score):
                            trial_score = float("inf")
                        if trial_score < float(best["selection_score"]):
                            best = {
                                "selection_score": trial_score,
                                "val_mae": val_mae,
                                "val_wape": val_wape,
                                "val_mdape": val_mdape,
                                "family": family,
                                "params": dict(params),
                                "best_iteration": meta.get("best_iteration"),
                            }
                    except Exception as e:
                        if mflow_on:
                            mlflow.set_tag("trial_status", "failed")
                            mlflow.set_tag("trial_error", str(e)[:500])
                        print(f"Trial {run_name} failed: {e}", file=sys.stderr)
                        traceback.print_exc()

        if not best["family"]:
            print("No successful tuning trials; aborting.", file=sys.stderr)
            return 1

        print(
            "Tuning champion: "
            f"{sel_mlflow}={float(best['selection_score']):.6f} "
            f"| val MAE $ {float(best['val_mae']):.4f} "
            f"| val WAPE {100.0 * float(best['val_wape']):.2f}% "
            f"| val median APE {100.0 * float(best['val_mdape']):.2f}%"
        )

        champion_family = str(best["family"])
        champion_params = dict(best["params"])
        champion_bi = best["best_iteration"]
        champion_run_id: str | None = None

        champion_ctx = (
            mlflow.start_run(nested=True, run_name="vinyliq_champion")
            if mflow_on
            else nullcontext()
        )
        with champion_ctx:
            if mflow_on:
                mlflow.log_param("model_family", champion_family)
                mlflow.log_param("selection_metric", sel_mlflow)
                mlflow.log_metric("best_selection_score", float(best["selection_score"]))
                for name, key in (
                    ("best_val_mae_dollars_approx", "val_mae"),
                    ("best_val_wape_dollars", "val_wape"),
                    ("best_val_median_ape_dollars", "val_mdape"),
                ):
                    bv = float(best[key])
                    if not math.isnan(bv):
                        mlflow.log_metric(name, bv)
                for pk, pv in champion_params.items():
                    mlflow.log_param(f"champion_hparam_{pk}", str(pv))

            champ = refit_champion(
                champion_family,
                champion_params,
                X_tr_full,
                y_tr_full,
                cols,
                best_iteration=champion_bi if isinstance(champion_bi, int) else None,
                random_state=seed,
                target_kind=target_kind,
            )
            pred_v = champ.predict_log1p(X_v)
            pred_test = champ.predict_log1p(X_test)
            y_test_lp = log1p_dollar_targets_for_metrics(y_test, med_test, target_kind)
            pred_test_lp = pred_log1p_dollar_for_metrics(pred_test, med_test, target_kind)
            test_mae = mae_dollars(y_test_lp, pred_test_lp)
            test_wape = wape_dollars(y_test_lp, pred_test_lp)
            test_mdape = median_ape_dollars(y_test_lp, pred_test_lp)
            y_tr_lp = log1p_dollar_targets_for_metrics(y_tr_full, med_tr_full, target_kind)
            test_mdape_bl = median_ape_train_median_baseline(y_tr_lp, y_test_lp)
            y_v_lp_q = log1p_dollar_targets_for_metrics(y_v, med_v, target_kind)
            pred_v_lp_q = pred_log1p_dollar_for_metrics(pred_v, med_v, target_kind)
            val_q = median_ape_dollar_quartiles(y_v_lp_q, pred_v_lp_q)
            test_q = median_ape_dollar_quartiles(y_test_lp, pred_test_lp)
            if mflow_on:
                for i, qv in enumerate(val_q):
                    if not math.isnan(qv):
                        mlflow.log_metric(
                            f"champion_val_median_ape_dollar_quartile_{i}", qv
                        )
                for i, qv in enumerate(test_q):
                    if not math.isnan(qv):
                        mlflow.log_metric(
                            f"champion_test_median_ape_dollar_quartile_{i}", qv
                        )
                mlflow.log_metric(
                    "test_median_ape_train_median_log_baseline", test_mdape_bl
                )
            qv_str = " | ".join(
                f"Q{i + 1} {100.0 * q:.1f}%"
                for i, q in enumerate(val_q)
                if not math.isnan(q)
            )
            qt_str = " | ".join(
                f"Q{i + 1} {100.0 * q:.1f}%"
                for i, q in enumerate(test_q)
                if not math.isnan(q)
            )
            print(
                f"Champion {champion_family} | holdout MAE $ {test_mae:.4f} | "
                f"WAPE {100.0 * test_wape:.2f}% | median APE {100.0 * test_mdape:.2f}% "
                f"| baseline median APE {100.0 * test_mdape_bl:.2f}%"
            )
            print(f"  Val median APE by true $ quartile (Q1=cheapest): {qv_str}")
            print(f"  Test median APE by true $ quartile: {qt_str}")
            if mflow_on:
                mlflow.log_metric("test_mae_dollars_approx", test_mae)
                mlflow.log_metric("test_wape_dollars", test_wape)
                mlflow.log_metric("test_median_ape_dollars", test_mdape)

            md.mkdir(parents=True, exist_ok=True)
            champ.save(md)
            _write_encoder_artifacts(md, catalog_encoders)
            tt_art = (
                {**(v.get("training_target") or {}), "kind": target_kind}
                if isinstance(v.get("training_target"), dict)
                else {"kind": target_kind}
            )
            _write_training_label_config(
                md,
                training_label_cfg,
                training_target=tt_art,
            )

            if mflow_art:
                if cfg_path.is_file():
                    mlflow.log_artifact(str(cfg_path), artifact_path="config")
                mlflow.log_artifacts(str(md), artifact_path="vinyliq_artifacts")

                arts = pyfunc_artifacts_dict(md)
                mlflow.pyfunc.log_model(
                    python_model=VinylIQPricePyFunc(),
                    artifacts=arts,
                    artifact_path="vinyliq_model",
                    input_example=build_pyfunc_input_example(target_kind=target_kind),
                )

            if mflow_on:
                ar = mlflow.active_run()
                if ar is not None:
                    champion_run_id = ar.info.run_id

        register = (
            bool(mlflow_cfg.get("register_best_model", True))
            and mflow_art
            and mflow_on
        )
        reg_name = str(mlflow_cfg.get("registry_model_name", "VinylIQPrice")).strip()
        staging_alias = str(mlflow_cfg.get("staging_alias", "staging")).strip()
        prod_alias = str(mlflow_cfg.get("production_alias", "production")).strip()
        promote = bool(mlflow_cfg.get("promote_production", False))

        if register and reg_name and champion_run_id and MlflowClient is not None:
            model_uri = f"runs:/{champion_run_id}/vinyliq_model"
            try:
                mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
                client = MlflowClient()
                if staging_alias:
                    client.set_registered_model_alias(
                        reg_name, staging_alias, mv.version
                    )
                if promote and prod_alias:
                    client.set_registered_model_alias(
                        reg_name, prod_alias, mv.version
                    )
                print(
                    f"Registered {reg_name} version {mv.version} from {model_uri} "
                    f"(alias {staging_alias!r})",
                )
            except Exception as e:
                print(f"Model registry skipped: {e}", file=sys.stderr)

    print(f"Saved model to {md}")
    return 0


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
    md = Path(paths.get("model_dir", root / "artifacts" / "vinyliq"))
    if not mp.is_absolute():
        mp = root / mp
    if not fs.is_absolute():
        fs = root / fs
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
    target_kind = training_target_kind_from_vinyliq(v)
    raw_tt = v.get("training_target")
    training_target_artifact: dict[str, object] = (
        {**raw_tt, "kind": target_kind}
        if isinstance(raw_tt, dict)
        else {"kind": target_kind}
    )
    print(
        "Training label: "
        f"mode={training_label_cfg.get('mode')!s}, "
        f"blend_median_weight={training_label_cfg.get('blend_median_weight')!s}, "
        f"spread_lowest_floor_ratio={training_label_cfg.get('spread_lowest_floor_ratio')!s}, "
        f"spread_min_median_weight={training_label_cfg.get('spread_min_median_weight')!s}, "
        f"spread_num_for_sale_reference={training_label_cfg.get('spread_num_for_sale_reference')!s}",
    )
    print(f"Training target kind: {target_kind}")
    z_clip = residual_z_clip_abs_from_vinyliq(v)
    if z_clip is not None:
        print(f"Residual z clip (abs): {z_clip}")

    Xrows, yvals, rids, catalog_encoders, median_anchors = load_training_frame(
        mp,
        fs,
        max_primary_artist_ids=max_art,
        max_primary_label_ids=max_lbl,
        training_label=training_label_cfg,
        training_target_kind=target_kind,
        residual_z_clip_abs=z_clip,
    )
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
                    "training_label_mode": str(training_label_cfg.get("mode", "median")),
                    "training_label_blend_median_weight": str(
                        training_label_cfg.get("blend_median_weight", 0.7)
                    ),
                    "training_label_spread_lowest_floor_ratio": str(
                        training_label_cfg.get("spread_lowest_floor_ratio")
                    ),
                    "training_label_spread_min_median_weight": str(
                        training_label_cfg.get("spread_min_median_weight")
                    ),
                    "training_label_spread_num_for_sale_reference": str(
                        training_label_cfg.get("spread_num_for_sale_reference")
                    ),
                }
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


if __name__ == "__main__":
    raise SystemExit(_cli_main())
