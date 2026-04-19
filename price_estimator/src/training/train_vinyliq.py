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
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from ..models.vinyliq_pyfunc import (
    VinylIQPricePyFunc,
    build_pyfunc_input_example,
    pyfunc_artifacts_dict,
)
from ..models.xgb_vinyliq import XGBVinylIQModel
from .label_synthesis import training_label_config_from_vinyliq
from .sale_floor_targets import (
    sale_floor_blend_bundle,
    sale_floor_blend_sf_cfg_for_policy,
)
from .search_space import sample_from_space
from .vinyliq_tuning_selection import (
    TrialRecord,
    _resolve_single_selection_metric,
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


def training_target_kind_from_vinyliq(v: dict | None) -> str:
    raw = (v or {}).get("training_target") or {}
    if not isinstance(raw, dict):
        raw = {}
    k = str(raw.get("kind", "residual_log_median")).strip().lower()
    if k in ("residual_log_median", "residual", "residual_log1p_median"):
        return TARGET_KIND_RESIDUAL_LOG_MEDIAN
    return TARGET_KIND_DOLLAR_LOG1P


def residual_z_clip_abs_from_vinyliq(v: dict | None) -> float | None:
    """Optional winsor on ``z = log1p(y_label) - log1p(m)`` to ``[-c, c]`` (null = off).

    ``c`` is a fixed half-width in **log1p-dollar residual space** (not tied to removed
    ``median_price`` columns).
    """
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


def _tuning_sample_weight_mode(v: dict[str, Any]) -> str | None:
    t = v.get("tuning") or {}
    sw = t.get("sample_weight")
    if sw is None:
        return None
    s = str(sw).strip()
    return s if s else None


def _format_sample_weight_multipliers(v: dict[str, Any]) -> dict[str, float] | None:
    t = v.get("tuning") or {}
    raw = t.get("format_sample_weight_multipliers")
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, float] = {}
    for k, val in raw.items():
        try:
            out[str(k)] = float(val)
        except (TypeError, ValueError):
            continue
    return out or None


def _training_label_console_summary(tl: dict[str, object]) -> str:
    """Human-readable label config: only keys that apply to ``mode``."""
    mode = str(tl.get("mode", "sale_floor_blend")).strip().lower()
    parts: list[str] = [f"mode={mode}"]
    if mode in ("sale_floor_blend", "sale_floor"):
        sfb = tl.get("sale_floor_blend")
        if isinstance(sfb, dict) and sfb:
            parts.append(
                "sale_floor_blend="
                + json.dumps(sfb, separators=(",", ":"), sort_keys=True)
            )
        parts.append(f"price_suggestion_grade(anchor)={tl.get('price_suggestion_grade')!s}")
    else:
        parts.append(
            "note=only sale_floor_blend / sale_floor are supported for VinylIQ training"
        )
    return ", ".join(parts)


def _training_label_mlflow_params(tl: dict[str, object]) -> dict[str, str]:
    """Flat params for MLflow (sale-floor training and optional nested knobs)."""
    mode = str(tl.get("mode", "sale_floor_blend")).strip().lower()
    out: dict[str, str] = {"training_label_mode": mode}
    if mode in ("sale_floor_blend", "sale_floor"):
        sfb = tl.get("sale_floor_blend")
        if isinstance(sfb, dict):
            for k, v in sorted(sfb.items()):
                out[f"training_label_sf_{k}"] = str(v)
        out["training_label_ps_grade_anchor"] = str(tl.get("price_suggestion_grade", ""))
    return out


def _mlflow_log_training_label_params(
    mlflow: Any,
    tl: dict[str, object],
) -> None:
    for k, v in _training_label_mlflow_params(tl).items():
        mlflow.log_param(k, v)


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


def _pick_newer_marketplace_row_dict(
    a: dict[str, object],
    b: dict[str, object],
) -> dict[str, object]:
    """Prefer latest ``fetched_at`` (ISO string order); tie-break by non-null field count."""
    fa = str(a.get("fetched_at") or "")
    fb = str(b.get("fetched_at") or "")
    if fa != fb:
        return a if fa > fb else b

    def score(d: dict[str, object]) -> int:
        return sum(1 for v in d.values() if v is not None)

    return a if score(a) >= score(b) else b


def _auto_top_k_id_encoder(n_labeled: int, n_unique: int) -> int:
    if n_unique <= 0:
        return 0
    if n_unique <= 500:
        return n_unique
    k = max(500, min(3000, n_labeled // 25))
    return min(k, n_unique)


def _load_sale_history_sidecars(
    sh_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """``release_sale`` rows and ``sale_history_fetch_status`` per ``release_id``."""
    sales: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fetch: dict[str, dict[str, Any]] = {}
    if not sh_path.is_file():
        return {}, {}
    conn = sqlite3.connect(str(sh_path))
    conn.row_factory = sqlite3.Row
    try:
        for r in conn.execute("SELECT * FROM release_sale"):
            d = dict(r)
            sales[str(d["release_id"])].append(d)
        for r in conn.execute("SELECT * FROM sale_history_fetch_status"):
            d = dict(r)
            fetch[str(d["release_id"])] = d
    finally:
        conn.close()
    return dict(sales), fetch


def _default_cold_start_flags(mx: dict[str, Any]) -> dict[str, float]:
    lo = mx.get("release_lowest_price") or mx.get("lowest_price") or mx.get("median_price")
    try:
        has_lf = 1.0 if lo is not None and float(lo) > 0 else 0.0
    except (TypeError, ValueError):
        has_lf = 0.0
    return {"has_sale_history": 0.0, "s_imputed": 0.0, "has_listing_floor": has_lf}


@dataclass(frozen=True)
class TrainingFrameLoad:
    """Rows, targets, and dual-policy sale-history flags from ``load_training_frame``.

    ``yvals_nm`` / ``yvals_ord`` may be NaN when that policy does not produce a dollar label;
    ensemble training fits each head only on finite targets, then both heads predict on every row
    (same as production pyfunc).
    """

    xrows: list[dict]
    yvals: list[float]
    rids: list[str]
    catalog_encoders: dict[str, dict[str, float]]
    median_anchors: list[float]
    has_nm_comp_sale: list[float]
    has_ord_comp_sale: list[float]
    yvals_nm: list[float]
    yvals_ord: list[float]


def _stored_target_from_dollar_label(
    y_dollar: float | None,
    m_anchor: float | None,
    *,
    training_target_kind: str,
    residual_z_clip_abs: float | None,
) -> float:
    if y_dollar is None or m_anchor is None:
        return float("nan")
    try:
        yf = float(y_dollar)
        mf = float(m_anchor)
    except (TypeError, ValueError):
        return float("nan")
    if yf <= 0 or mf <= 0:
        return float("nan")
    if training_target_kind == TARGET_KIND_RESIDUAL_LOG_MEDIAN:
        z = float(np.log1p(yf) - np.log1p(mf))
        if residual_z_clip_abs is not None and residual_z_clip_abs > 0:
            z = float(np.clip(z, -residual_z_clip_abs, residual_z_clip_abs))
        return z
    return float(np.log1p(yf))


def _blend_sweep_pairs_from_ensemble_dict(
    raw: dict[str, Any],
    *,
    default_t: float,
    default_s: float,
) -> list[tuple[float, float]] | None:
    """
    Optional Cartesian grid or explicit ``pairs`` for post-hoc val selection of ``(t, s)``.

    Returns ``None`` when sweep is disabled; otherwise a non-empty list of ``(t, s)`` pairs.
    """
    sw = raw.get("blend_sweep")
    if not isinstance(sw, dict) or not sw.get("enabled", False):
        return None
    if "pairs" in sw:
        expl = sw.get("pairs")
        if not isinstance(expl, list):
            raise ValueError("ensemble.blend_sweep.pairs must be a list")
        if not expl:
            raise ValueError("ensemble.blend_sweep.pairs is empty")
        out: list[tuple[float, float]] = []
        for row in expl:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                out.append((float(row[0]), float(row[1])))
        if not out:
            raise ValueError("ensemble.blend_sweep.pairs has no valid [t, s] rows")
        return out
    ts = sw.get("t")
    ss = sw.get("s")
    if isinstance(ts, (list, tuple)) and isinstance(ss, (list, tuple)):
        if not ts or not ss:
            return [(default_t, default_s)]
        return [(float(t), float(s)) for t in ts for s in ss]
    return [(default_t, default_s)]


def ensemble_blend_config_from_vinyliq(v: dict[str, Any] | None) -> dict[str, Any] | None:
    raw = (v or {}).get("ensemble")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    blend = raw.get("blend") or {}
    kind = str(blend.get("kind", "log_anchor_sigmoid")).strip().lower()
    if kind != "log_anchor_sigmoid":
        raise ValueError(
            f"Unsupported vinyliq.ensemble.blend.kind {kind!r} (only log_anchor_sigmoid)"
        )
    dt = float(blend.get("t", 4.0))
    ds = float(blend.get("s", 0.35))
    sweep_pairs = _blend_sweep_pairs_from_ensemble_dict(
        raw, default_t=dt, default_s=ds
    )
    return {
        "kind": kind,
        "t": dt,
        "s": ds,
        "share_hparams": bool(raw.get("share_hparams", True)),
        "blend_sweep_pairs": sweep_pairs,
    }


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


def _save_ensemble_manifest_and_estimators(
    model_dir: Path,
    *,
    backend: str,
    target_kind: str,
    target_was_log1p: bool,
    feature_columns: list[str],
    champ_nm: Any,
    champ_ord: Any,
    blend_t: float,
    blend_s: float,
) -> None:
    """Write schema_version 3 manifest + per-head estimator joblibs for pyfunc."""
    import joblib

    model_dir.mkdir(parents=True, exist_ok=True)
    nm_path = "regressor_ensemble_nm.joblib"
    ord_path = "regressor_ensemble_ord.joblib"
    joblib.dump(champ_nm.estimator, model_dir / nm_path)
    joblib.dump(champ_ord.estimator, model_dir / ord_path)
    # Legacy artifact name (ordinal head); unused by ensemble pyfunc but keeps layouts consistent.
    joblib.dump(champ_ord.estimator, model_dir / "regressor.joblib")
    joblib.dump(feature_columns, model_dir / "feature_columns.joblib")
    joblib.dump(
        target_kind == TARGET_KIND_DOLLAR_LOG1P and target_was_log1p,
        model_dir / "target_log1p.joblib",
    )
    manifest = {
        "schema_version": 3,
        "backend": backend,
        "target_kind": target_kind,
        "ensemble": {
            "enabled": True,
            "blend": {
                "kind": "log_anchor_sigmoid",
                "t": float(blend_t),
                "s": float(blend_s),
            },
            "regressor_nm": nm_path,
            "regressor_ord": ord_path,
        },
    }
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))


def _fit_frequency_capped_id_encoder(ids: list[str], max_k: int) -> dict[str, float]:
    if max_k <= 0:
        return {}
    c = Counter(i for i in ids if i)
    if not c:
        return {}
    top = [pid for pid, _ in c.most_common(max_k)]
    return {pid: float(i + 1) for i, pid in enumerate(top)}


def _catalog_encoders_from_saved_bundle(
    saved: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    Rebuild the in-memory ``catalog_encoders`` dict from ``catalog_encoders.json`` in a model dir.

    Ensures the four feature maps exist and values are float; copies ``_id_encoder_meta`` when
    present so diagnostics match training.
    """
    out: dict[str, Any] = {}
    for key in ("genre", "country", "primary_artist_id", "primary_label_id"):
        raw = saved.get(key)
        if isinstance(raw, dict):
            out[key] = {str(kk): float(vv) for kk, vv in raw.items()}
        else:
            out[key] = {}
    meta = saved.get("_id_encoder_meta")
    if isinstance(meta, dict):
        out["_id_encoder_meta"] = {str(k): float(v) for k, v in meta.items()}
    return out  # type: ignore[return-value]


def load_training_frame(
    marketplace_db: Path,
    feature_store_db: Path,
    *,
    max_primary_artist_ids: int | None = None,
    max_primary_label_ids: int | None = None,
    training_label: dict[str, object] | None = None,
    training_target_kind: str = TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    residual_z_clip_abs: float | None = None,
    sale_history_db: Path | None = None,
    catalog_encoders_override: dict[str, Any] | None = None,
) -> TrainingFrameLoad:
    tl = training_label or training_label_config_from_vinyliq({})
    mode_l = str(tl.get("mode", "sale_floor_blend")).strip().lower()
    if mode_l not in ("sale_floor_blend", "sale_floor"):
        raise ValueError(
            f"Unsupported training_label.mode {mode_l!r} for VinylIQ training "
            "(expected sale_floor_blend or sale_floor)."
        )
    sales_by_rid: dict[str, list[dict[str, Any]]] = {}
    fetch_by_rid: dict[str, dict[str, Any]] = {}
    if mode_l in ("sale_floor_blend", "sale_floor"):
        if sale_history_db is None or not Path(sale_history_db).is_file():
            print(
                "Warning: sale_floor_blend needs sale_history.sqlite "
                "(vinyliq.paths.sale_history_db); sold nowcast s will be missing "
                "unless listing floor / PS-only paths yield y.",
                file=sys.stderr,
            )
        else:
            sales_by_rid, fetch_by_rid = _load_sale_history_sidecars(Path(sale_history_db))

    year_by_rid: dict[str, Any] = {}
    if Path(feature_store_db).is_file():
        conn_y = sqlite3.connect(str(feature_store_db))
        conn_y.row_factory = sqlite3.Row
        try:
            for r in conn_y.execute("SELECT release_id, year FROM releases_features"):
                year_by_rid[str(r["release_id"])] = r["year"]
        finally:
            conn_y.close()

    conn_m = sqlite3.connect(str(marketplace_db))
    conn_m.row_factory = sqlite3.Row
    cur = conn_m.execute(
        """
        SELECT release_id, fetched_at, median_price, lowest_price, num_for_sale,
               price_suggestions_json, release_lowest_price, release_num_for_sale,
               community_want, community_have, blocked_from_sale
        FROM marketplace_stats
        WHERE (
            COALESCE(release_lowest_price, lowest_price, median_price) IS NOT NULL
            AND COALESCE(release_lowest_price, lowest_price, median_price) > 0
        ) OR (
            price_suggestions_json IS NOT NULL
            AND TRIM(price_suggestions_json) != ''
            AND TRIM(price_suggestions_json) != '{}'
        )
        """
    )
    by_rid: dict[str, dict[str, object]] = {}
    for r in cur.fetchall():
        rid = str(r["release_id"])
        rd = {k: r[k] for k in r.keys()}
        prev = by_rid.get(rid)
        if prev is None:
            by_rid[rid] = rd
        else:
            by_rid[rid] = _pick_newer_marketplace_row_dict(prev, rd)

    labels: dict[str, float] = {}
    medians: dict[str, float] = {}
    labels_nm: dict[str, float] = {}
    medians_nm: dict[str, float] = {}
    labels_ord: dict[str, float] = {}
    medians_ord: dict[str, float] = {}
    has_nm_sale_by_rid: dict[str, float] = {}
    has_ord_sale_by_rid: dict[str, float] = {}
    marketplace_extra: dict[str, dict[str, Any]] = {}
    row_cold_flags: dict[str, dict[str, float]] = {}
    ps_grade = str(tl.get("price_suggestion_grade") or "Near Mint (NM or M-)").strip()
    sf_cfg = tl.get("sale_floor_blend") if isinstance(tl.get("sale_floor_blend"), dict) else {}
    sf_nm = sale_floor_blend_sf_cfg_for_policy(sf_cfg, "nm_substrings_only")
    sf_ord = sale_floor_blend_sf_cfg_for_policy(sf_cfg, "ordinal_cascade")
    primary_pol = str(sf_cfg.get("sale_condition_policy", "nm_substrings_only")).strip().lower()
    if primary_pol not in ("nm_substrings_only", "ordinal_cascade"):
        primary_pol = "nm_substrings_only"

    for rid, rd in by_rid.items():
        yr_raw = year_by_rid.get(rid)
        release_year: float | None
        try:
            release_year = float(yr_raw) if yr_raw is not None else None
        except (TypeError, ValueError):
            release_year = None
        if release_year is not None and not math.isfinite(release_year):
            release_year = None
        rd_d = dict(rd)
        sales = sales_by_rid.get(rid, [])
        fetch = fetch_by_rid.get(rid)
        out_nm = sale_floor_blend_bundle(
            rd_d,
            sales,
            fetch,
            sf_cfg=sf_nm,
            nm_grade_key=ps_grade,
            release_year=release_year,
        )
        out_ord = sale_floor_blend_bundle(
            rd_d,
            sales,
            fetch,
            sf_cfg=sf_ord,
            nm_grade_key=ps_grade,
            release_year=release_year,
        )
        yn, mn, fn = out_nm
        yo, mo, fo = out_ord
        has_nm_sale_by_rid[rid] = float(fn.get("has_sale_history", 0.0))
        has_ord_sale_by_rid[rid] = float(fo.get("has_sale_history", 0.0))

        if yn is not None and yn > 0:
            mnu = (
                float(mn)
                if mn is not None and float(mn) > 0
                else float(yn)
            )
            labels_nm[rid] = float(yn)
            medians_nm[rid] = mnu
        if yo is not None and yo > 0:
            mou = (
                float(mo)
                if mo is not None and float(mo) > 0
                else float(yo)
            )
            labels_ord[rid] = float(yo)
            medians_ord[rid] = mou

        y_p, m_p, flags_p = out_ord if primary_pol == "ordinal_cascade" else out_nm
        if flags_p:
            row_cold_flags[rid] = flags_p
        if y_p is not None and y_p > 0:
            m_use = (
                float(m_p)
                if m_p is not None and float(m_p) > 0
                else float(y_p)
            )
            labels[rid] = float(y_p)
            medians[rid] = m_use
            marketplace_extra[rid] = {
                "community_want": rd.get("community_want"),
                "community_have": rd.get("community_have"),
                "release_num_for_sale": rd.get("release_num_for_sale"),
                "lowest_price": rd.get("lowest_price"),
                "median_price": rd.get("median_price"),
                "release_lowest_price": rd.get("release_lowest_price"),
                "num_for_sale": rd.get("num_for_sale"),
                "blocked_from_sale": rd.get("blocked_from_sale"),
            }
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
        mx = marketplace_extra.get(rid, {})
        r_cat = dict(r)
        labeled.append(r_cat)

    artist_ids_per_row = [first_artist_id(r) for r in labeled]
    label_ids_per_row = [first_label_id(r) for r in labeled]
    n_art_u = len({x for x in artist_ids_per_row if x})
    n_lbl_u = len({x for x in label_ids_per_row if x})
    n_lab = len(labeled)

    if catalog_encoders_override is not None:
        catalog_encoders = _catalog_encoders_from_saved_bundle(catalog_encoders_override)
    else:
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

        catalog_encoders = {
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
    has_nm_comp_sale: list[float] = []
    has_ord_comp_sale: list[float] = []
    yvals_nm: list[float] = []
    yvals_ord: list[float] = []
    for r in labeled:
        rid = str(r.get("release_id", ""))
        y_dollar = labels[rid]
        mp = medians[rid]
        gidx = g2i.get(str(r.get("genre") or "").strip().lower(), 0.0)
        cidx = c2i.get(str(r.get("country") or "").strip().lower(), 0.0)
        aidx = a2i.get(first_artist_id(r), 0.0)
        lbidx = l2i.get(first_label_id(r), 0.0)
        mx = marketplace_extra.get(rid, {})
        stats = {
            "median_price": mx.get("median_price"),
            "lowest_price": mx.get("lowest_price"),
            "release_lowest_price": mx.get("release_lowest_price"),
            "num_for_sale": mx.get("num_for_sale"),
            "community_want": mx.get("community_want"),
            "community_have": mx.get("community_have"),
            "release_num_for_sale": mx.get("release_num_for_sale"),
            "blocked_from_sale": mx.get("blocked_from_sale"),
        }
        cf = row_cold_flags.get(rid) or _default_cold_start_flags(mx)
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
            cold_start_flags=cf,
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
        has_nm_comp_sale.append(has_nm_sale_by_rid.get(rid, 0.0))
        has_ord_comp_sale.append(has_ord_sale_by_rid.get(rid, 0.0))
        y_nm = _stored_target_from_dollar_label(
            labels_nm.get(rid),
            medians_nm.get(rid),
            training_target_kind=training_target_kind,
            residual_z_clip_abs=residual_z_clip_abs,
        )
        y_ord = _stored_target_from_dollar_label(
            labels_ord.get(rid),
            medians_ord.get(rid),
            training_target_kind=training_target_kind,
            residual_z_clip_abs=residual_z_clip_abs,
        )
        yvals_nm.append(y_nm)
        yvals_ord.append(y_ord)

    return TrainingFrameLoad(
        xrows=Xrows,
        yvals=yvals,
        rids=rids,
        catalog_encoders=catalog_encoders,
        median_anchors=median_anchors,
        has_nm_comp_sale=has_nm_comp_sale,
        has_ord_comp_sale=has_ord_comp_sale,
        yvals_nm=yvals_nm,
        yvals_ord=yvals_ord,
    )


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
        f"log1p(m_anchor): std={float(np.std(log1pm)):.5f}  "
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
    Legacy single-metric resolution (``composite`` falls back to MdAPE here).

    Prefer ``parse_selection_objective`` for full tuning behavior.
    """
    raw = str((tuning or {}).get("selection_metric", "median_ape")).strip().lower()
    if raw == "composite":
        return ("mdape", "val_median_ape_dollars")
    return _resolve_single_selection_metric(tuning)


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


if __name__ == "__main__":
    raise SystemExit(_cli_main())
