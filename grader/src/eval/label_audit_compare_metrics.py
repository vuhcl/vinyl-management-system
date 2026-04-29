from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_BEFORE_PATH = Path("grader/reports/label_audit_baseline_reference.json")


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metric_rows(
    before: dict[str, Any], after: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("val", "test"):
        b_split = (before.get("eval") or {}).get(split) or {}
        a_split = (after.get("eval") or {}).get(split) or {}
        for target in ("sleeve", "media"):
            b_m = (b_split.get(target) or {}).get("macro_f1")
            a_m = (a_split.get(target) or {}).get("macro_f1")
            if b_m is None or a_m is None:
                continue
            out.append(
                {
                    "split": split,
                    "target": target,
                    "macro_f1_before": float(b_m),
                    "macro_f1_after": float(a_m),
                    "delta_macro_f1": float(a_m) - float(b_m),
                }
            )
    return out


def _headline_delta(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    before_vals = [float(r["macro_f1_before"]) for r in rows]
    after_vals = [float(r["macro_f1_after"]) for r in rows]
    mean_before = sum(before_vals) / len(before_vals)
    mean_after = sum(after_vals) / len(after_vals)
    abs_delta = mean_after - mean_before
    rel_pct = (abs_delta / mean_before * 100.0) if mean_before > 0 else 0.0
    return {
        "mean_macro_f1_before": mean_before,
        "mean_macro_f1_after": mean_after,
        "delta_macro_f1": abs_delta,
        "delta_percent": rel_pct,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compare baseline eval metrics before vs after "
            "patch replay."
        )
    )
    p.add_argument(
        "--before",
        default=str(DEFAULT_BEFORE_PATH),
        help=(
            "Path to pre-patch run JSON. "
            f"Defaults to {DEFAULT_BEFORE_PATH}."
        ),
    )
    p.add_argument(
        "--after", required=True, help="Path to post-patch run JSON."
    )
    p.add_argument(
        "--output",
        default="grader/reports/label_audit_metric_delta.json",
        help="Output JSON summary path.",
    )
    args = p.parse_args()
    before_path = Path(args.before)
    if not before_path.is_file():
        raise FileNotFoundError(
            f"Before file not found: {before_path}. "
            "Pass --before or create baseline reference JSON first."
        )
    before = _load(before_path)
    after = _load(Path(args.after))
    rows = _metric_rows(before, after)
    headline = _headline_delta(rows)
    summary = {
        "before": str(args.before),
        "after": str(args.after),
        "headline": headline,
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if headline is not None:
        print(
            "Resume bullet X (mean macro-F1 relative gain): "
            f"{headline['delta_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
