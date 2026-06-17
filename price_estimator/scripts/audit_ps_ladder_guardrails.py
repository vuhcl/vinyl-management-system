#!/usr/bin/env python3
"""Audit PS ladder guardrail gate triggers on marketplace + sale_history data.

Computes reference floors under two sources:
- **inference:** Discogs ``sale_stats_*`` overlay fields (usually absent in SQLite).
- **training:** sale_history quartiles (all USD sales before ``t_ref``) + credible listing.

Emits ratio percentiles, gate trigger counts, and top-N examples per bucket.

Note: frozen SQLite rows may not match live Discogs PS ladders (e.g. demo release
12830828 had a post-collection outlier sale). Use corpus percentiles for gate
calibration; use ``--release-ids`` for spot checks against *current* live stats
when validating inflation scenarios.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from price_estimator.src.inference.anchor_guardrails import (
    trim_price_suggestions_json,
)
from price_estimator.src.inference.anchor_guardrails_config import (
    AnchorGuardrailsConfig,
    anchor_guardrails_config_from_vinyliq,
)
from price_estimator.src.sale_floor.reference_floor import (
    gate_outcomes_for_ref,
    reference_floor_inference_usd,
    reference_floor_training_usd,
)
from price_estimator.src.storage.marketplace_db import marketplace_inference_stats_from_row
from price_estimator.src.training.label_synthesis import training_label_config_from_vinyliq
from price_estimator.src.training.train_vinyliq.training_frame import _load_sale_history_sidecars


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(root: Path, p: Path | None, fallback: str) -> Path:
    x = p or Path(fallback)
    return x if x.is_absolute() else root / x


def _percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    arr = sorted(vals)
    if len(arr) == 1:
        return float(arr[0])
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    w = pos - lo
    return float(arr[lo] * (1.0 - w) + arr[hi] * w)


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw.get("vinyliq") or {}


def _mp_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(r[1]) for r in conn.execute("PRAGMA table_info(marketplace_stats)")}


def _attach_sale_stats_from_row(row: sqlite3.Row, cols: set[str]) -> dict[str, Any]:
    d = {k: row[k] for k in row.keys()}
    for key in (
        "sale_stats_average_usd",
        "sale_stats_median_usd",
        "sale_stats_high_usd",
        "sale_stats_low_usd",
    ):
        if key in cols:
            d[key] = row[key]
    return d


def _trim_would_change(
    stats: dict[str, Any],
    cfg: AnchorGuardrailsConfig,
    *,
    nm_grade_key: str,
) -> bool:
    copy_stats = copy.deepcopy(stats)
    return trim_price_suggestions_json(
        copy_stats, cfg, nm_grade_key=nm_grade_key
    )


def _summarize_ratios(ratios: list[float]) -> dict[str, float | None | int]:
    if not ratios:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "n": len(ratios),
        "p50": _percentile(ratios, 0.50),
        "p90": _percentile(ratios, 0.90),
        "p99": _percentile(ratios, 0.99),
        "max": max(ratios),
    }


def _audit_row(
    mp_row: dict[str, Any],
    sale_rows: list[dict[str, Any]],
    fetch_status: dict[str, Any] | None,
    *,
    ag_cfg: AnchorGuardrailsConfig,
    nm_grade_key: str,
) -> dict[str, Any]:
    stats = marketplace_inference_stats_from_row(mp_row)
    ref_inf = reference_floor_inference_usd(stats, ag_cfg, nm_grade_key=nm_grade_key)
    ref_train, train_diag = reference_floor_training_usd(
        mp_row,
        sale_rows,
        fetch_status,
        nm_grade_key=nm_grade_key,
        ag_cfg=ag_cfg,
    )
    gates_inf = gate_outcomes_for_ref(
        ref_inf, stats, ag_cfg, nm_grade_key=nm_grade_key
    )
    stats_train = copy.deepcopy(stats)
    for key in (
        "sale_stats_low_usd",
        "sale_stats_median_usd",
        "sale_stats_average_usd",
        "sale_stats_high_usd",
        "n_sales",
    ):
        val = train_diag.get(key)
        if val is not None:
            stats_train[key] = val
    gates_train = gate_outcomes_for_ref(
        ref_train, stats_train, ag_cfg, nm_grade_key=nm_grade_key
    )
    trim = _trim_would_change(stats, ag_cfg, nm_grade_key=nm_grade_key)
    has_overlay = any(
        stats.get(k) is not None
        for k in ("sale_stats_median_usd", "sale_stats_average_usd")
    )
    return {
        "release_id": str(mp_row.get("release_id") or ""),
        "has_sale_stats_overlay": has_overlay,
        "reference_floor_inference_usd": ref_inf,
        "reference_floor_training_usd": ref_train,
        "training_diag": train_diag,
        "inference_gates": gates_inf,
        "training_gates": gates_train,
        "trim_would_change": trim,
    }


def _bucket_key(row: dict[str, Any], source: str) -> str:
    g = row[f"{source}_gates"]
    if not g.get("guardrails_active"):
        return "no_reference"
    strength = float(g.get("blend_strength") or 0.0)
    if strength > 0.0 or g.get("sale_stats_blend_apply"):
        return "blend_on"
    if g.get("is_inflated_ladder"):
        return "inflated_no_blend"
    return "normal"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("price_estimator/configs/base.yaml"),
    )
    ap.add_argument("--marketplace-db", type=Path, default=None)
    ap.add_argument("--sale-history-db", type=Path, default=None)
    ap.add_argument(
        "--release-ids",
        type=Path,
        default=None,
        help="Optional newline-separated release IDs (one per line)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Random sample size (default: all rows with PS ladder)",
    )
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument(
        "--format",
        choices=("text", "json", "tsv"),
        default="text",
    )
    args = ap.parse_args()

    root = _pkg_root()
    v = _load_config(args.config)
    paths = v.get("paths") or {}
    mp_path = _resolve_path(
        root,
        args.marketplace_db,
        str(paths.get("marketplace_db", "data/cache/marketplace_stats.sqlite")),
    )
    sh_path = _resolve_path(
        root,
        args.sale_history_db,
        str(paths.get("sale_history_db", "data/cache/sale_history.sqlite")),
    )
    if not mp_path.is_file():
        print(f"Missing marketplace db: {mp_path}", file=sys.stderr)
        return 1

    tl = training_label_config_from_vinyliq(v)
    nm_grade_key = str(tl.get("price_suggestion_grade") or "Near Mint (NM or M-)")

    ag_cfg = anchor_guardrails_config_from_vinyliq(v)
    ag_raw = v.get("anchor_guardrails")
    if not isinstance(ag_raw, dict):
        inf = v.get("inference") or {}
        ag_raw = inf.get("anchor_guardrails") if isinstance(inf.get("anchor_guardrails"), dict) else {}

    release_ids: set[str] | None = None
    if args.release_ids is not None and args.release_ids.is_file():
        release_ids = {
            ln.strip()
            for ln in args.release_ids.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }

    sales_by_rid: dict[str, list[dict[str, Any]]] = {}
    fetch_by_rid: dict[str, dict[str, Any]] = {}
    if sh_path.is_file():
        sales_by_rid, fetch_by_rid = _load_sale_history_sidecars(sh_path)

    conn = sqlite3.connect(mp_path)
    mp_cols = _mp_columns(conn)
    has_sale_stats_cols = "sale_stats_median_usd" in mp_cols
    extra_cols = ", ".join(
        c
        for c in (
            "sale_stats_average_usd",
            "sale_stats_median_usd",
            "sale_stats_high_usd",
            "sale_stats_low_usd",
        )
        if c in mp_cols
    )
    conn.row_factory = sqlite3.Row
    if release_ids:
        placeholders = ",".join("?" for _ in release_ids)
        sql = f"""
            SELECT release_id, fetched_at, release_lowest_price, num_for_sale,
                   price_suggestions_json{', ' + extra_cols if extra_cols else ''}
            FROM marketplace_stats
            WHERE release_id IN ({placeholders})
        """
        mp_rows = list(conn.execute(sql, tuple(sorted(release_ids))))
    elif args.sample is not None and args.sample > 0:
        sql = f"""
            SELECT release_id, fetched_at, release_lowest_price, num_for_sale,
                   price_suggestions_json{', ' + extra_cols if extra_cols else ''}
            FROM marketplace_stats
            WHERE price_suggestions_json IS NOT NULL
              AND TRIM(price_suggestions_json) NOT IN ('', '{{}}')
            ORDER BY RANDOM()
            LIMIT ?
        """
        mp_rows = list(conn.execute(sql, (int(args.sample),)))
    else:
        sql = f"""
            SELECT release_id, fetched_at, release_lowest_price, num_for_sale,
                   price_suggestions_json{', ' + extra_cols if extra_cols else ''}
            FROM marketplace_stats
            WHERE price_suggestions_json IS NOT NULL
              AND TRIM(price_suggestions_json) NOT IN ('', '{{}}')
        """
        mp_rows = list(conn.execute(sql))
    conn.close()

    audited: list[dict[str, Any]] = []
    overlay_populated = 0
    for row in mp_rows:
        mp_dict = _attach_sale_stats_from_row(row, mp_cols)
        rid = str(mp_dict["release_id"])
        out = _audit_row(
            mp_dict,
            sales_by_rid.get(rid, []),
            fetch_by_rid.get(rid),
            ag_cfg=ag_cfg,
            nm_grade_key=nm_grade_key,
        )
        if out["has_sale_stats_overlay"]:
            overlay_populated += 1
        audited.append(out)

    n = len(audited)
    ratios_inf: list[float] = []
    ratios_train: list[float] = []
    buckets_inf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    buckets_train: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in audited:
        gi = row["inference_gates"]
        gt = row["training_gates"]
        if gi.get("ratio_mx_ref") is not None:
            ratios_inf.append(float(gi["ratio_mx_ref"]))
        if gt.get("ratio_mx_ref") is not None:
            ratios_train.append(float(gt["ratio_mx_ref"]))
        buckets_inf[_bucket_key(row, "inference")].append(row)
        buckets_train[_bucket_key(row, "training")].append(row)

    def _count_gates(source: str) -> dict[str, int]:
        prefix = f"{source}_gates"
        return {
            "guardrails_active": sum(
                1 for r in audited if r[prefix]["guardrails_active"]
            ),
            "is_inflated_ladder": sum(
                1 for r in audited if r[prefix]["is_inflated_ladder"]
            ),
            "sale_stats_blend_apply": sum(
                1 for r in audited if r[prefix]["sale_stats_blend_apply"]
            ),
            "blend_strength_positive": sum(
                1
                for r in audited
                if float(r[prefix].get("blend_strength") or 0.0) > 0.0
            ),
            "ratio_blend_fallback": sum(
                1 for r in audited if r[prefix].get("ratio_blend_fallback")
            ),
            "trim_would_change": sum(1 for r in audited if r["trim_would_change"]),
        }

    summary = {
        "n_rows_audited": n,
        "marketplace_db": str(mp_path),
        "sale_history_db": str(sh_path) if sh_path.is_file() else None,
        "nm_grade_key": nm_grade_key,
        "anchor_guardrails": ag_raw,
        "sale_stats_columns_in_sqlite": has_sale_stats_cols,
        "rows_with_sale_stats_overlay_values": overlay_populated,
        "overlay_coverage_pct": round(100.0 * overlay_populated / n, 2) if n else 0.0,
        "ratio_mx_ref_inference": _summarize_ratios(ratios_inf),
        "ratio_mx_ref_training": _summarize_ratios(ratios_train),
        "gate_counts_inference": _count_gates("inference"),
        "gate_counts_training": _count_gates("training"),
        "bucket_counts_inference": {k: len(v) for k, v in buckets_inf.items()},
        "bucket_counts_training": {k: len(v) for k, v in buckets_train.items()},
    }

    top_n = max(1, int(args.top_n))
    examples: dict[str, Any] = {}
    for source, buckets in (("inference", buckets_inf), ("training", buckets_train)):
        examples[source] = {}
        for bucket, rows in sorted(buckets.items()):
            ranked = sorted(
                rows,
                key=lambda r: float(
                    r[f"{source}_gates"].get("ratio_mx_ref") or 0.0
                ),
                reverse=True,
            )[:top_n]
            examples[source][bucket] = [
                {
                    "release_id": r["release_id"],
                    "ratio_mx_ref": r[f"{source}_gates"].get("ratio_mx_ref"),
                    "reference_floor_usd": r[f"reference_floor_{source}_usd"],
                    "mx_usd": r[f"{source}_gates"].get("mx_usd"),
                    "nm_rung_usd": r[f"{source}_gates"].get("nm_rung_usd"),
                    "blend_strength": r[f"{source}_gates"].get("blend_strength"),
                    "R_sale": r[f"{source}_gates"].get("R_sale"),
                    "R_ladder": r[f"{source}_gates"].get("R_ladder"),
                    "blend_direction": r[f"{source}_gates"].get("blend_direction"),
                    "ratio_blend_fallback": r[f"{source}_gates"].get(
                        "ratio_blend_fallback"
                    ),
                }
                for r in ranked
            ]

    payload = {"summary": summary, "examples": examples}
    if args.format == "json":
        print(json.dumps(payload, indent=2, default=str))
    elif args.format == "tsv":
        print(
            "release_id\tref_inference\tref_training\tmx_ratio_inf\tmx_ratio_train\t"
            "inflated_inf\tblend_inf\tinflated_train\tblend_train\ttrim"
        )
        for r in audited:
            gi = r["inference_gates"]
            gt = r["training_gates"]
            print(
                f"{r['release_id']}\t{r['reference_floor_inference_usd']}\t"
                f"{r['reference_floor_training_usd']}\t{gi.get('ratio_mx_ref')}\t"
                f"{gt.get('ratio_mx_ref')}\t{gi.get('is_inflated_ladder')}\t"
                f"{gi.get('sale_stats_blend_apply')}\t{gt.get('is_inflated_ladder')}\t"
                f"{gt.get('sale_stats_blend_apply')}\t{r['trim_would_change']}"
            )
    else:
        print("=== PS ladder guardrail audit ===")
        print(json.dumps(summary, indent=2, default=str))
        print("\n--- Top examples by bucket (training ref) ---")
        for bucket, rows in examples.get("training", {}).items():
            print(f"\n[{bucket}]")
            for ex in rows:
                ratio = ex.get("ratio_mx_ref")
                ratio_s = f"{ratio:.3f}" if ratio is not None else "n/a"
                strength = ex.get("blend_strength")
                strength_s = f"{strength:.2f}" if strength is not None else "n/a"
                print(
                    f"  {ex['release_id']}: mx/ref={ratio_s} "
                    f"ref={ex['reference_floor_usd']} mx={ex['mx_usd']} "
                    f"nm={ex['nm_rung_usd']} strength={strength_s} "
                    f"R_sale={ex.get('R_sale')} R_ladder={ex.get('R_ladder')} "
                    f"dir={ex.get('blend_direction')}"
                )
        print("\n--- Gate policy note ---")
        print(
            "Ratio blend: continuous strength from R_ladder vs R_sale "
            f"(full at excess ~{ag_cfg.ratio_blend_full_strength}); "
            f"legacy fallback uses inflated_max_rung_to_reference="
            f"{ag_cfg.inflated_max_rung_to_reference} when avg missing."
        )
        if not has_sale_stats_cols or overlay_populated == 0:
            print(
                "\nInference ref mostly unavailable in SQLite "
                "(sale_stats_* not stored); use training ref counts for label-time gates."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
