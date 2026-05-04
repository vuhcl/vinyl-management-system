"""Multi-trial tuning loop and related diagnostics."""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ...features.vinyliq_features import (
    default_feature_columns,
    residual_training_feature_columns,
    row_dict_for_inference,
)
from ...mlflow_tracking import configure_mlflow_from_config
from ...models.condition_adjustment import default_params, save_params
from ...models.fitted_regressor import (
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    combine_anchor_and_format_sample_weights,
    ensemble_blend_weight_log_anchor,
    fit_regressor,
    log1p_dollar_targets_for_metrics,
    mae_dollars,
    median_ape_dollars,
    median_ape_dollar_quartiles,
    median_ape_quartile_format_slice_diagnostics,
    median_ape_quartile_format_slice_table,
    median_ape_train_median_baseline,
    metrics_dollar_from_log1p_masked,
    pred_log1p_dollar_for_metrics,
    refit_champion,
    wape_dollars,
    weighted_format_median_ape_dollars,
)
from ...models.vinyliq_pyfunc import (
    VinylIQPricePyFunc,
    build_pyfunc_input_example,
    pyfunc_artifacts_dict,
)
from ...models.xgb_vinyliq import XGBVinylIQModel
from ..label_synthesis import training_label_config_from_vinyliq
from ..search_space import sample_from_space
from ..vinyliq_tuning_selection import (
    TrialRecord,
    base_selection_score,
    build_cv_fold_val_release_sets,
    build_trial_record,
    log_split_anchor_format_diagnostics,
    parse_selection_format_weights,
    parse_selection_objective,
    parse_tuning_constraints,
    pick_champion_trial,
    row_masks_from_release_sets,
)
from .catalog_encoders import _write_encoder_artifacts
from .ensemble_manifest import _save_ensemble_manifest_and_estimators
from .release_train_split import train_test_split_by_release
from .training_config import (
    _config_path_for_mlflow,
    _enabled_families,
    _mlflow_flags,
    _mlflow_log_training_label_params,
    _resolve_tuning_selection_metric,
    _root,
    _training_label_mlflow_params,
    _write_training_label_config,
)


def _log_slice_metrics_block(
    *,
    split_label: str,
    y_lp: np.ndarray,
    pred_lp: np.ndarray,
    mask_nm: np.ndarray,
    mask_cold: np.ndarray,
    mask_ord: np.ndarray,
    mflow_on: bool,
    mlflow: Any,
    min_count: int = 15,
) -> None:
    """NM-comps, cold-start (no NM comps), and ordinal-comps slices in log1p-dollar space."""
    for name, mask in (
        ("nm_comps", mask_nm),
        ("cold_start_no_nm_comps", mask_cold),
        ("ordinal_comps", mask_ord),
    ):
        mae_s, wape_s, mdape_s = metrics_dollar_from_log1p_masked(
            y_lp, pred_lp, mask, min_count=min_count
        )
        n_m = int(np.sum(mask & np.isfinite(y_lp) & np.isfinite(pred_lp)))
        if math.isnan(mdape_s):
            print(
                f"  {split_label} {name}: n<{min_count} (n={n_m}) — MdAPE skipped",
            )
        else:
            print(
                f"  {split_label} {name}: MAE ${mae_s:.4f} | "
                f"WAPE {100.0 * wape_s:.2f}% | median APE {100.0 * mdape_s:.2f}% "
                f"(n={n_m})",
            )
        if mflow_on:
            mlflow.log_metric(f"{split_label}_{name}_n_rows", float(n_m))
            if not math.isnan(mae_s):
                mlflow.log_metric(f"{split_label}_{name}_mae_dollars_approx", mae_s)
            if not math.isnan(wape_s):
                mlflow.log_metric(f"{split_label}_{name}_wape_dollars", wape_s)
            if not math.isnan(mdape_s):
                mlflow.log_metric(f"{split_label}_{name}_median_ape_dollars", mdape_s)

def _slice_metric_debug_enabled() -> bool:
    return os.environ.get("VINYLIQ_SLICE_METRIC_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _log_quartile_format_slice_diagnostics(
    split_label: str,
    y_lp: np.ndarray,
    pred_lp: np.ndarray,
    X_sub: np.ndarray,
    cols: list[str],
) -> None:
    """Stderr table: median / mean / p90 / max APE when ``VINYLIQ_SLICE_METRIC_DEBUG`` is set."""
    rows = median_ape_quartile_format_slice_diagnostics(
        y_lp, pred_lp, X_sub, cols, min_count=15
    )
    print(
        f"[VINYLIQ_SLICE_METRIC_DEBUG] {split_label}: quartile×format "
        "(MdAPE / mean / p90 / max as % of true $)",
        file=sys.stderr,
    )
    for r in rows:
        md, mn, p9, mx = (
            r["median_ape"],
            r["mean_ape"],
            r["p90_ape"],
            r["max_ape"],
        )
        print(
            f"  Q{r['quartile'] + 1} {r['slice']:9s} n={r['n_rows']:<5d} "
            f"md={100.0 * md:7.4f}% mean={100.0 * mn:7.4f}% "
            f"p90={100.0 * p9:7.4f}% max={100.0 * mx:7.4f}%",
            file=sys.stderr,
        )

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
    has_nm_comp_sale: np.ndarray,
    has_ord_comp_sale: np.ndarray,
    y_nm: np.ndarray,
    y_ord: np.ndarray,
    ensemble_cfg: dict[str, Any] | None,
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

    h_nm = np.asarray(has_nm_comp_sale, dtype=np.float64).ravel()
    h_ord = np.asarray(has_ord_comp_sale, dtype=np.float64).ravel()
    y_nm_all = np.asarray(y_nm, dtype=np.float64).ravel()
    y_ord_all = np.asarray(y_ord, dtype=np.float64).ravel()
    if not (len(h_nm) == len(rids) == len(h_ord) == len(y_nm_all) == len(y_ord_all)):
        raise ValueError("Per-row policy arrays must align with rids")

    cons = parse_tuning_constraints(tuning)
    sel_obj = parse_selection_objective(tuning)
    sel_mlflow = sel_obj.mlflow_name
    sel_fmt_weights = parse_selection_format_weights(tuning)
    wf_min_count = int(tuning.get("selection_format_min_count", 15))
    cv_folds_cfg = int(tuning.get("cv_folds", 5))
    cv_agg = str(tuning.get("cv_agg", "mean")).strip().lower()
    if cv_agg not in ("mean", "max"):
        cv_agg = "mean"
    cv_strat_raw = tuning.get("cv_stratify")
    cv_stratify = (
        "anchor_quartile"
        if str(cv_strat_raw).strip().lower() == "anchor_quartile"
        else None
    )
    use_cv = cv_folds_cfg > 1 and len(train_r) >= 2
    if use_cv:
        eff_kv = min(cv_folds_cfg, len(train_r))
        fold_val_sets = build_cv_fold_val_release_sets(
            train_r,
            rids,
            med,
            eff_kv,
            int(seed) + 2,
            stratify=cv_stratify,  # type: ignore[arg-type]
        )
        cv_folds_effective = len(fold_val_sets)
    else:
        fold_val_sets = []
        cv_folds_effective = 1

    log_split_anchor_format_diagnostics(
        train_mask, test_mask, med, X_all, cols
    )

    spaces = v.get("search_spaces") or {}
    families = _enabled_families(v)
    sw_mode = _tuning_sample_weight_mode(v)
    fmt_mults = _format_sample_weight_multipliers(v)
    sw_tt = combine_anchor_and_format_sample_weights(
        med_tt, sw_mode, X_tt, cols, fmt_mults
    )
    sw_full = combine_anchor_and_format_sample_weights(
        med_tr_full, sw_mode, X_tr_full, cols, fmt_mults
    )

    trial_records: list[TrialRecord] = []

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
            _mlflow_log_training_label_params(mlflow, training_label_cfg)
            mlflow.log_param("n_train_outer", int(train_mask.sum()))
            mlflow.log_param("n_test_outer", int(test_mask.sum()))
            mlflow.log_param("n_tune_train", int((tune_train_mask & train_mask).sum()))
            mlflow.log_param("n_tune_val", int((val_mask & train_mask).sum()))
            mlflow.log_param("tuning_selection_metric", sel_mlflow)
            mlflow.log_param("tuning_selection_metric_raw", str(tuning.get("selection_metric", "")))
            mlflow.log_param("cv_folds_configured", str(cv_folds_cfg))
            mlflow.log_param("cv_folds_effective", str(cv_folds_effective))
            mlflow.log_param("cv_use_release_cv", str(use_cv))
            mlflow.log_param("cv_agg", cv_agg)
            mlflow.log_param("cv_stratify", str(cv_stratify or "random_shuffle"))
            if ensemble_cfg:
                mlflow.log_param("ensemble_enabled", "true")
                mlflow.log_param("ensemble_blend_t", str(ensemble_cfg["t"]))
                mlflow.log_param("ensemble_blend_s", str(ensemble_cfg["s"]))
            else:
                mlflow.log_param("ensemble_enabled", "false")
            mlflow.log_param("constraints_enabled", str(cons.enabled))
            if cons.enabled:
                mlflow.log_param("constraints_mdape_max", str(cons.mdape_max))
                mlflow.log_param("constraints_wape_max", str(cons.wape_max))
                mlflow.log_param("constraints_violation_fallback", cons.violation_fallback)
            if sw_mode:
                mlflow.log_param("tuning_sample_weight", sw_mode)
            if fmt_mults:
                mlflow.log_param(
                    "tuning_format_sample_weight_multipliers",
                    json.dumps(fmt_mults, sort_keys=True),
                )
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
        if sw_mode:
            print(f"Tuning sample_weight mode: {sw_mode}")
        if fmt_mults:
            print(f"Tuning format_sample_weight_multipliers: {fmt_mults}")

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
                        mdapes: list[float] = []
                        wf_mdapes: list[float] | None = (
                            [] if sel_obj.use_weighted_format_mdape else None
                        )
                        maes: list[float] = []
                        wapes: list[float] = []
                        best_iters: list[int | None] = []
                        if use_cv:
                            for val_rel in fold_val_sets:
                                tt_m, va_m = row_masks_from_release_sets(
                                    rids, train_r, val_rel
                                )
                                X_tt_f = X_all[tt_m]
                                y_tt_f = y_all[tt_m]
                                X_v_f = X_all[va_m]
                                y_v_f = y_all[va_m]
                                if X_tt_f.shape[0] < 5 or X_v_f.shape[0] < 1:
                                    continue
                                m_tt_f = med[tt_m]
                                m_v_f = med[va_m]
                                sw_f = combine_anchor_and_format_sample_weights(
                                    m_tt_f,
                                    sw_mode,
                                    X_tt_f,
                                    cols,
                                    fmt_mults,
                                )
                                reg, meta = fit_regressor(
                                    family,
                                    params,
                                    X_tt_f,
                                    y_tt_f,
                                    cols,
                                    X_val=X_v_f,
                                    y_val=y_v_f,
                                    early_stopping_rounds=es_int,
                                    random_state=seed,
                                    target_kind=target_kind,
                                    sample_weight=sw_f,
                                )
                                pred_vf = reg.predict_log1p(X_v_f)
                                y_v_lp_m = log1p_dollar_targets_for_metrics(
                                    y_v_f, m_v_f, target_kind
                                )
                                pred_v_lp = pred_log1p_dollar_for_metrics(
                                    pred_vf, m_v_f, target_kind
                                )
                                mdapes.append(
                                    median_ape_dollars(y_v_lp_m, pred_v_lp)
                                )
                                if wf_mdapes is not None:
                                    wf_mdapes.append(
                                        weighted_format_median_ape_dollars(
                                            y_v_lp_m,
                                            pred_v_lp,
                                            X_v_f,
                                            cols,
                                            sel_fmt_weights,
                                            min_count=wf_min_count,
                                        )
                                    )
                                maes.append(mae_dollars(y_v_lp_m, pred_v_lp))
                                wapes.append(wape_dollars(y_v_lp_m, pred_v_lp))
                                best_iters.append(meta.get("best_iteration"))
                            if not mdapes:
                                raise RuntimeError("no valid CV folds for trial")
                        else:
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
                                sample_weight=sw_tt,
                            )
                            pred_v = reg.predict_log1p(X_v)
                            y_v_lp_m = log1p_dollar_targets_for_metrics(
                                y_v, med_v, target_kind
                            )
                            pred_v_lp = pred_log1p_dollar_for_metrics(
                                pred_v, med_v, target_kind
                            )
                            mdapes.append(
                                median_ape_dollars(y_v_lp_m, pred_v_lp)
                            )
                            if wf_mdapes is not None:
                                wf_mdapes.append(
                                    weighted_format_median_ape_dollars(
                                        y_v_lp_m,
                                        pred_v_lp,
                                        X_v,
                                        cols,
                                        sel_fmt_weights,
                                        min_count=wf_min_count,
                                    )
                                )
                            maes.append(mae_dollars(y_v_lp_m, pred_v_lp))
                            wapes.append(wape_dollars(y_v_lp_m, pred_v_lp))
                            best_iters.append(meta.get("best_iteration"))

                        rec = build_trial_record(
                            family=family,
                            params=dict(params),
                            mdapes=mdapes,
                            maes=maes,
                            wapes=wapes,
                            best_iters=best_iters,
                            cv_agg=cv_agg,  # type: ignore[arg-type]
                            cons=cons,
                            sel_obj=sel_obj,
                            cv_folds_used=len(mdapes),
                            selection_mdapes=wf_mdapes,
                        )
                        if rec is None:
                            raise RuntimeError("CV metrics non-finite")
                        if mflow_on:
                            mlflow.log_metric(
                                "val_mae_dollars_approx", rec.val_mae
                            )
                            mlflow.log_metric("val_wape_dollars", rec.val_wape)
                            mlflow.log_metric(
                                "val_median_ape_dollars", rec.val_mdape
                            )
                            mlflow.log_metric(
                                "val_base_objective", rec.base_score
                            )
                            mlflow.log_metric(
                                "trial_feasible", 1.0 if rec.feasible else 0.0
                            )
                            mlflow.log_metric(
                                "trial_violation_slack", rec.slack
                            )
                            mlflow.log_metric(
                                "trial_penalty_objective", rec.pen_score
                            )
                            mlflow.log_metric(
                                "tuning_cv_folds_per_trial", float(rec.cv_folds_used)
                            )
                            if rec.best_iteration is not None:
                                mlflow.log_metric(
                                    "best_iteration", float(rec.best_iteration)
                                )
                        trial_records.append(rec)
                    except Exception as e:
                        if mflow_on:
                            mlflow.set_tag("trial_status", "failed")
                            mlflow.set_tag("trial_error", str(e)[:500])
                        print(f"Trial {run_name} failed: {e}", file=sys.stderr)
                        traceback.print_exc()

        if not trial_records:
            print("No successful tuning trials; aborting.", file=sys.stderr)
            return 1

        best_rec, pick_reason = pick_champion_trial(trial_records, cons)
        if best_rec is None:
            print(
                "Constraint violation_fallback=abort and no feasible trials "
                "(or no trials). Aborting.",
                file=sys.stderr,
            )
            return 1

        best: dict[str, object] = {
            "selection_score": best_rec.base_score,
            "val_mae": best_rec.val_mae,
            "val_wape": best_rec.val_wape,
            "val_mdape": best_rec.val_mdape,
            "family": best_rec.family,
            "params": best_rec.params,
            "best_iteration": best_rec.best_iteration,
            "_pick_reason": pick_reason,
        }

        cv_note = (
            f"cv_folds={best_rec.cv_folds_used} agg={cv_agg}"
            if use_cv
            else "cv_folds=1 (inner split)"
        )
        print(
            "Tuning champion: "
            f"{sel_mlflow}={float(best['selection_score']):.6f} "
            f"| val MAE $ {float(best['val_mae']):.4f} "
            f"| val WAPE {100.0 * float(best['val_wape']):.2f}% "
            f"| val median APE {100.0 * float(best['val_mdape']):.2f}% "
            f"| pick={pick_reason} | {cv_note}"
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
                mlflow.set_tag(
                    "champion_pick_reason", str(best.get("_pick_reason", ""))
                )
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

            if ensemble_cfg:
                blend_t = float(ensemble_cfg["t"])
                blend_s = float(ensemble_cfg["s"])
                y_nm_tr = y_nm_all[train_mask]
                y_ord_tr = y_ord_all[train_mask]
                m_nm_tr = np.isfinite(y_nm_tr)
                m_ord_tr = np.isfinite(y_ord_tr)
                n_nm_fit = int(np.sum(m_nm_tr))
                n_ord_fit = int(np.sum(m_ord_tr))
                if n_nm_fit < 20 or n_ord_fit < 20:
                    print(
                        "Ensemble: need >=20 outer-train rows per head with a valid "
                        f"policy label (NM={n_nm_fit}, Ord={n_ord_fit}).",
                        file=sys.stderr,
                    )
                    return 1
                sw_nm = (
                    sw_full[m_nm_tr]
                    if sw_full is not None
                    else None
                )
                sw_ord = (
                    sw_full[m_ord_tr]
                    if sw_full is not None
                    else None
                )
                champ_nm = refit_champion(
                    champion_family,
                    champion_params,
                    X_tr_full[m_nm_tr],
                    y_nm_tr[m_nm_tr],
                    cols,
                    best_iteration=champion_bi if isinstance(champion_bi, int) else None,
                    random_state=seed,
                    target_kind=target_kind,
                    sample_weight=sw_nm,
                )
                champ_ord = refit_champion(
                    champion_family,
                    champion_params,
                    X_tr_full[m_ord_tr],
                    y_ord_tr[m_ord_tr],
                    cols,
                    best_iteration=champion_bi if isinstance(champion_bi, int) else None,
                    random_state=seed,
                    target_kind=target_kind,
                    sample_weight=sw_ord,
                )
                pred_v_nm = champ_nm.predict_log1p(X_v)
                pred_v_ord = champ_ord.predict_log1p(X_v)
                pred_test_nm = champ_nm.predict_log1p(X_test)
                pred_test_ord = champ_ord.predict_log1p(X_test)
                pred_v_lp_nm = pred_log1p_dollar_for_metrics(
                    pred_v_nm, med_v, target_kind
                )
                pred_v_lp_ord = pred_log1p_dollar_for_metrics(
                    pred_v_ord, med_v, target_kind
                )
                y_v_lp_primary = log1p_dollar_targets_for_metrics(
                    y_v, med_v, target_kind
                )
                sweep_pairs = ensemble_cfg.get("blend_sweep_pairs")
                if sweep_pairs is not None:
                    best_sc = float("inf")
                    best_pair: tuple[float, float] = (blend_t, blend_s)
                    for t_try, s_try in sweep_pairs:
                        w_try = ensemble_blend_weight_log_anchor(
                            med_v,
                            center_log1p=float(t_try),
                            scale=float(s_try),
                        )
                        pred_try = (
                            w_try * pred_v_lp_nm
                            + (1.0 - w_try) * pred_v_lp_ord
                        )
                        if sel_obj.use_weighted_format_mdape:
                            mdape_v = weighted_format_median_ape_dollars(
                                y_v_lp_primary,
                                pred_try,
                                X_v,
                                cols,
                                sel_fmt_weights,
                                min_count=wf_min_count,
                            )
                        else:
                            mdape_v = median_ape_dollars(
                                y_v_lp_primary, pred_try
                            )
                        ma = mae_dollars(y_v_lp_primary, pred_try)
                        wa = wape_dollars(y_v_lp_primary, pred_try)
                        sc = base_selection_score(sel_obj, mdape_v, ma, wa)
                        if not math.isfinite(sc):
                            continue
                        if sc < best_sc:
                            best_sc = sc
                            best_pair = (float(t_try), float(s_try))
                    if math.isfinite(best_sc):
                        blend_t, blend_s = best_pair
                    print(
                        "  Ensemble blend sweep (val, "
                        f"objective={sel_obj.mlflow_name}): best t={blend_t:g} "
                        f"s={blend_s:g} — {len(sweep_pairs)} (t,s) grid, "
                        f"n_val={len(y_v)}"
                    )
                    if mflow_on:
                        mlflow.log_param("ensemble_blend_selected_t", str(blend_t))
                        mlflow.log_param("ensemble_blend_selected_s", str(blend_s))
                        if math.isfinite(best_sc):
                            mlflow.log_metric(
                                "ensemble_blend_sweep_val_selection_score",
                                float(best_sc),
                            )
                w_v = ensemble_blend_weight_log_anchor(
                    med_v, center_log1p=blend_t, scale=blend_s
                )
                pred_v_lp = w_v * pred_v_lp_nm + (1.0 - w_v) * pred_v_lp_ord

                pred_test_lp_nm = pred_log1p_dollar_for_metrics(
                    pred_test_nm, med_test, target_kind
                )
                pred_test_lp_ord = pred_log1p_dollar_for_metrics(
                    pred_test_ord, med_test, target_kind
                )
                w_te = ensemble_blend_weight_log_anchor(
                    med_test, center_log1p=blend_t, scale=blend_s
                )
                pred_test_lp = w_te * pred_test_lp_nm + (1.0 - w_te) * pred_test_lp_ord

                y_test_lp_nm_h = log1p_dollar_targets_for_metrics(
                    y_nm_all[test_mask], med_test, target_kind
                )
                y_test_lp_ord_h = log1p_dollar_targets_for_metrics(
                    y_ord_all[test_mask], med_test, target_kind
                )
                m_nm_te = np.isfinite(y_nm_all[test_mask])
                m_ord_te = np.isfinite(y_ord_all[test_mask])
                if int(np.sum(m_nm_te)) >= 1:
                    test_mdape_nm_h = median_ape_dollars(
                        y_test_lp_nm_h[m_nm_te], pred_test_lp_nm[m_nm_te]
                    )
                else:
                    test_mdape_nm_h = float("nan")
                if int(np.sum(m_ord_te)) >= 1:
                    test_mdape_ord_h = median_ape_dollars(
                        y_test_lp_ord_h[m_ord_te], pred_test_lp_ord[m_ord_te]
                    )
                else:
                    test_mdape_ord_h = float("nan")
                nm_s = (
                    f"{100.0 * test_mdape_nm_h:.2f}%"
                    if not math.isnan(test_mdape_nm_h)
                    else "n/a"
                )
                ord_s = (
                    f"{100.0 * test_mdape_ord_h:.2f}%"
                    if not math.isnan(test_mdape_ord_h)
                    else "n/a"
                )
                print(
                    "  Ensemble heads (test, vs own label where that label exists): "
                    f"NM median APE {nm_s} (n={int(np.sum(m_nm_te))}) | "
                    f"Ord median APE {ord_s} (n={int(np.sum(m_ord_te))})"
                )
                if mflow_on:
                    if not math.isnan(test_mdape_nm_h):
                        mlflow.log_metric(
                            "champion_test_median_ape_nm_head_own_label",
                            test_mdape_nm_h,
                        )
                    if not math.isnan(test_mdape_ord_h):
                        mlflow.log_metric(
                            "champion_test_median_ape_ord_head_own_label",
                            test_mdape_ord_h,
                        )
            else:
                champ = refit_champion(
                    champion_family,
                    champion_params,
                    X_tr_full,
                    y_tr_full,
                    cols,
                    best_iteration=champion_bi if isinstance(champion_bi, int) else None,
                    random_state=seed,
                    target_kind=target_kind,
                    sample_weight=sw_full,
                )
                pred_v = champ.predict_log1p(X_v)
                pred_test = champ.predict_log1p(X_test)
                pred_v_lp = pred_log1p_dollar_for_metrics(pred_v, med_v, target_kind)
                pred_test_lp = pred_log1p_dollar_for_metrics(
                    pred_test, med_test, target_kind
                )

            y_test_lp = log1p_dollar_targets_for_metrics(y_test, med_test, target_kind)
            test_mae = mae_dollars(y_test_lp, pred_test_lp)
            test_wape = wape_dollars(y_test_lp, pred_test_lp)
            test_mdape = median_ape_dollars(y_test_lp, pred_test_lp)
            y_tr_lp = log1p_dollar_targets_for_metrics(y_tr_full, med_tr_full, target_kind)
            test_mdape_bl = median_ape_train_median_baseline(y_tr_lp, y_test_lp)
            y_v_lp_q = log1p_dollar_targets_for_metrics(y_v, med_v, target_kind)
            pred_v_lp_q = pred_v_lp
            val_q = median_ape_dollar_quartiles(y_v_lp_q, pred_v_lp_q)
            test_q = median_ape_dollar_quartiles(y_test_lp, pred_test_lp)
            slice_val = median_ape_quartile_format_slice_table(
                y_v_lp_q, pred_v_lp_q, X_v, cols, min_count=15
            )
            slice_test = median_ape_quartile_format_slice_table(
                y_test_lp, pred_test_lp, X_test, cols, min_count=15
            )
            if _slice_metric_debug_enabled():
                _log_quartile_format_slice_diagnostics(
                    "val", y_v_lp_q, pred_v_lp_q, X_v, cols
                )
                _log_quartile_format_slice_diagnostics(
                    "test", y_test_lp, pred_test_lp, X_test, cols
                )
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
                for r in slice_val:
                    slice_mdape = float(r["median_ape"])
                    if not math.isnan(slice_mdape):
                        mlflow.log_metric(
                            f"champion_val_q{int(r['quartile']) + 1}_fmt_{r['slice']}_mdape",
                            slice_mdape,
                        )
                        mlflow.log_metric(
                            f"champion_val_q{int(r['quartile']) + 1}_fmt_{r['slice']}_n",
                            float(r["n_rows"]),
                        )
                for r in slice_test:
                    slice_mdape = float(r["median_ape"])
                    if not math.isnan(slice_mdape):
                        mlflow.log_metric(
                            f"champion_test_q{int(r['quartile']) + 1}_fmt_{r['slice']}_mdape",
                            slice_mdape,
                        )
                        mlflow.log_metric(
                            f"champion_test_q{int(r['quartile']) + 1}_fmt_{r['slice']}_n",
                            float(r["n_rows"]),
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
            blend_note = " (blend vs primary label)" if ensemble_cfg else ""
            print(
                f"Champion {champion_family}{blend_note} | holdout MAE $ {test_mae:.4f} | "
                f"WAPE {100.0 * test_wape:.2f}% | median APE {100.0 * test_mdape:.2f}% "
                f"| baseline median APE {100.0 * test_mdape_bl:.2f}%"
            )
            print(f"  Val median APE by true $ quartile (Q1=cheapest): {qv_str}")
            print(f"  Test median APE by true $ quartile: {qt_str}")
            for label, srows in (
                ("Val MdAPE by quartile × format", slice_val),
                ("Test MdAPE by quartile × format", slice_test),
            ):
                for qi in range(4):
                    bits: list[str] = []
                    for r in srows:
                        if int(r["quartile"]) != qi:
                            continue
                        slice_mdape = float(r["median_ape"])
                        if math.isnan(slice_mdape):
                            continue
                        bits.append(
                            f"{r['slice']} {100.0 * slice_mdape:.1f}% (n={r['n_rows']})"
                        )
                    if bits:
                        print(f"  {label} Q{qi + 1}: " + " | ".join(bits))
            h_nm_val = h_nm[val_mask & train_mask]
            h_ord_val = h_ord[val_mask & train_mask]
            h_nm_test = h_nm[test_mask]
            h_ord_test = h_ord[test_mask]
            print("  Val slices (NM-comps / cold-start / ordinal-comps):")
            _log_slice_metrics_block(
                split_label="val",
                y_lp=y_v_lp_q,
                pred_lp=pred_v_lp,
                mask_nm=h_nm_val > 0.5,
                mask_cold=h_nm_val <= 0.5,
                mask_ord=h_ord_val > 0.5,
                mflow_on=mflow_on,
                mlflow=mlflow,
            )
            print("  Test slices (NM-comps / cold-start / ordinal-comps):")
            _log_slice_metrics_block(
                split_label="test",
                y_lp=y_test_lp,
                pred_lp=pred_test_lp,
                mask_nm=h_nm_test > 0.5,
                mask_cold=h_nm_test <= 0.5,
                mask_ord=h_ord_test > 0.5,
                mflow_on=mflow_on,
                mlflow=mlflow,
            )
            if mflow_on:
                mlflow.log_metric("test_mae_dollars_approx", test_mae)
                mlflow.log_metric("test_wape_dollars", test_wape)
                mlflow.log_metric("test_median_ape_dollars", test_mdape)

            md.mkdir(parents=True, exist_ok=True)
            if ensemble_cfg:
                _save_ensemble_manifest_and_estimators(
                    md,
                    backend=champ_nm.backend,
                    target_kind=champ_nm.target_kind,
                    target_was_log1p=(
                        champ_nm.target_kind == TARGET_KIND_DOLLAR_LOG1P
                        and champ_nm.target_was_log1p
                    ),
                    feature_columns=cols,
                    champ_nm=champ_nm,
                    champ_ord=champ_ord,
                    blend_t=float(blend_t),
                    blend_s=float(blend_s),
                )
            else:
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

