#!/usr/bin/env python3
"""
Emit ``grade_delta_scale.json`` for ``load_params_with_grade_delta_overlays``.

Default: pooled cross-grade fit from ``release_sale``, marketplace anchor USD,
and ``releases_features.year``. Use ``--placeholder`` for bootstrap JSON without DB reads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from price_estimator.src.models.grade_delta_scale_fit import (
    fit_grade_delta_scale_from_frame,
    sale_frame_from_dbs,
)
from price_estimator.src.models.grade_delta_scale_schema import (
    build_placeholder_grade_delta_fit,
    write_grade_delta_scale_json,
)


def _parse_float_list(s: str | None) -> tuple[float, ...] | None:
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return tuple(float(x) for x in parts)


def main() -> None:
    pe_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Emit grade_delta_scale.json (v1 schema).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSON path")
    p.add_argument(
        "--placeholder",
        action="store_true",
        help="Write bootstrap JSON (no DB reads)",
    )
    p.add_argument("--price-ref-usd", type=float, default=50.0)
    p.add_argument("--price-gamma", type=float, default=0.0)
    p.add_argument("--age-k", type=float, default=0.0)
    p.add_argument(
        "--sale-history-db",
        type=Path,
        default=pe_root / "data" / "cache" / "sale_history.sqlite",
    )
    p.add_argument(
        "--marketplace-db",
        type=Path,
        default=pe_root / "data" / "cache" / "marketplace_stats.sqlite",
    )
    p.add_argument(
        "--feature-store-db",
        type=Path,
        default=pe_root / "data" / "feature_store.sqlite",
    )
    p.add_argument(
        "--nm-grade-key",
        type=str,
        default="Near Mint (NM or M-)",
        help="Price suggestion grade key (anchor ladder fallback)",
    )
    p.add_argument("--min-bin-rows", type=int, default=30)
    p.add_argument("--min-grade-rows", type=int, default=5)
    p.add_argument("--base-alpha", type=float, default=0.06)
    p.add_argument("--base-beta", type=float, default=0.04)
    p.add_argument(
        "--no-fit-alpha-beta",
        action="store_true",
        help="Keep base-alpha/base-beta fixed; only grid-search price_gamma and age_k (legacy).",
    )
    p.add_argument(
        "--beta-per-alpha-fallback",
        type=float,
        default=None,
        help="When asymmetric strata are too sparse, split s=α+β with β=(ratio)*α (default: base-beta/base-alpha).",
    )
    p.add_argument("--price-scale-min", type=float, default=0.25)
    p.add_argument("--price-scale-max", type=float, default=4.0)
    p.add_argument(
        "--gamma-grid",
        type=str,
        default=None,
        help="Comma-separated price_gamma values (default: built-in grid)",
    )
    p.add_argument(
        "--age-k-grid",
        type=str,
        default=None,
        help="Comma-separated age_k values (default: built-in grid)",
    )
    args = p.parse_args()

    if args.placeholder:
        blob = build_placeholder_grade_delta_fit(
            price_ref_usd=args.price_ref_usd,
            price_gamma=args.price_gamma,
            age_k=args.age_k,
        )
        write_grade_delta_scale_json(args.out, blob)
        print(f"Wrote placeholder {args.out}")
        return

    for label, path in (
        ("sale_history", args.sale_history_db),
        ("marketplace", args.marketplace_db),
        ("feature_store", args.feature_store_db),
    ):
        if not path.is_file():
            print(f"Missing {label} DB: {path}", file=sys.stderr)
            sys.exit(2)

    df = sale_frame_from_dbs(
        args.sale_history_db,
        args.marketplace_db,
        args.feature_store_db,
        nm_grade_key=args.nm_grade_key,
    )
    try:
        blob = fit_grade_delta_scale_from_frame(
            df,
            nm_grade_key=args.nm_grade_key,
            base_alpha=args.base_alpha,
            base_beta=args.base_beta,
            fit_alpha_beta=not args.no_fit_alpha_beta,
            beta_per_alpha_fallback=args.beta_per_alpha_fallback,
            min_bin_rows=args.min_bin_rows,
            min_grade_rows=args.min_grade_rows,
            price_scale_min=args.price_scale_min,
            price_scale_max=args.price_scale_max,
            gammas=_parse_float_list(args.gamma_grid),
            age_ks=_parse_float_list(args.age_k_grid),
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)

    write_grade_delta_scale_json(args.out, blob)
    meta = blob["fit_metadata"]
    bins = meta.get("bins_used")
    ab = ""
    if "fitted_alpha" in meta:
        ab = f", alpha={meta['fitted_alpha']:.5g}, beta={meta['fitted_beta']:.5g}"
    print(f"Wrote {args.out} (rows={len(df)}, bins={bins}{ab})")


if __name__ == "__main__":
    main()
