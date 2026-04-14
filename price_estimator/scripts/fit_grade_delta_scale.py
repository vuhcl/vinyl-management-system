#!/usr/bin/env python3
"""
Emit ``grade_delta_scale.json`` for ``load_params_with_grade_delta_overlays``.

Extend this entrypoint with sale-history joins and pooled fits when data coverage allows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from price_estimator.src.models.grade_delta_scale_schema import (
    build_placeholder_grade_delta_fit,
    write_grade_delta_scale_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Emit grade_delta_scale.json (v1 schema).")
    p.add_argument("--out", type=Path, required=True, help="Output JSON path")
    p.add_argument("--price-ref-usd", type=float, default=50.0)
    p.add_argument("--price-gamma", type=float, default=0.0)
    p.add_argument("--age-k", type=float, default=0.0)
    args = p.parse_args()

    blob = build_placeholder_grade_delta_fit(
        price_ref_usd=args.price_ref_usd,
        price_gamma=args.price_gamma,
        age_k=args.age_k,
    )
    write_grade_delta_scale_json(args.out, blob)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
