"""Diff two ``rule_engine_baseline.json`` snapshots.

Small helper that supports the Track-B iteration loop documented in
``grader/README.md``. Given two baseline JSON files produced by
``_write_rule_engine_baseline`` (see ``grader/src/pipeline.py``), it reports:

* For each split / target, ``delta_macro_f1`` and ``delta_accuracy``
  (model-to-rule-adjusted deltas, same as the JSON).
* For each rule-owned grade (``Poor``, ``Generic`` by default), the delta in
  ``recall_adjusted`` (and ``recall_model``) under ``slice_recall``.
* For each rule-owned ``by_after`` bucket, the delta in ``override_precision``
  and the raw n_helpful / n_harmful / n_neutral counts (when present).

The on-disk format is ``{ "captured_at", "commit", "splits": { split: { target: ... } } }``.

The tool deliberately only prints; no policy (pass/fail) is baked in. The
acceptance thresholds live in ``grader/README.md`` "Rule-engine iteration
loop" so they can be tightened without touching code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt_delta(before: float | None, after: float | None) -> str:
    if before is None and after is None:
        return "n/a"
    if before is None:
        return f"  new   -> {after:.4f}" if after is not None else "n/a"
    if after is None:
        return f"{before:.4f} -> removed"
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{before:.4f} -> {after:.4f} ({sign}{delta:.4f})"


def _splits_dict(snap: Mapping[str, Any]) -> dict[str, Any]:
    """
    Canonical layer: ``splits[split_name][target]``
    (see ``_write_rule_engine_baseline``).
    """
    s = snap.get("splits")
    if isinstance(s, dict) and s:
        return s
    # Legacy: ``targets[target][split]`` (older experiments)
    out: dict[str, dict[str, Any]] = {}
    targets = snap.get("targets")
    if isinstance(targets, dict):
        for target, by_split in targets.items():
            if not isinstance(by_split, dict):
                continue
            for split, payload in by_split.items():
                out.setdefault(split, {})[target] = payload
    return out


def _walk_split_target(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for layer in (_splits_dict(before), _splits_dict(after)):
        for split, by_target in layer.items():
            if not isinstance(by_target, dict):
                continue
            for target in by_target:
                pairs.add((str(split), str(target)))
    return sorted(pairs)


def _bucket_keys(
    before_bucket: Mapping[str, Any] | None,
    after_bucket: Mapping[str, Any] | None,
) -> list[str]:
    keys: set[str] = set()
    if before_bucket:
        keys.update(before_bucket.keys())
    if after_bucket:
        keys.update(after_bucket.keys())
    return sorted(keys)


def diff_baselines(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    rule_owned: tuple[str, ...] = ("Poor", "Generic"),
) -> str:
    out: list[str] = []
    out.append("rule_engine_baseline.json diff")
    b_commit = before.get("commit", before.get("git_sha", "unknown"))
    a_commit = after.get("commit", after.get("git_sha", "unknown"))
    b_time = before.get("captured_at", before.get("timestamp", ""))
    a_time = after.get("captured_at", after.get("timestamp", ""))
    out.append(f"  before commit: {b_commit}  @ {b_time}")
    out.append(f"  after  commit: {a_commit}  @ {a_time}")
    out.append("")

    sb = _splits_dict(before)
    sa = _splits_dict(after)

    for split, target in _walk_split_target(before, after):
        before_t = (sb.get(split) or {}).get(target) or {}
        after_t = (sa.get(split) or {}).get(target) or {}
        if not before_t and not after_t:
            continue
        out.append(f"[{split} | {target}]")
        for metric in ("delta_macro_f1", "delta_accuracy"):
            out.append(
                f"  {metric:28s} "
                f"{_fmt_delta(before_t.get(metric), after_t.get(metric))}"
            )

        before_slice = before_t.get("slice_recall", {}) or {}
        after_slice = after_t.get("slice_recall", {}) or {}
        for grade in rule_owned:
            b = before_slice.get(grade, {}) or {}
            a = after_slice.get(grade, {}) or {}
            for metric in ("recall_adjusted", "recall_model"):
                out.append(
                    f"  {grade}.{metric:22s} "
                    f"{_fmt_delta(b.get(metric), a.get(metric))}"
                )

        before_by_after = before_t.get("by_after", {}) or {}
        after_by_after = after_t.get("by_after", {}) or {}
        for grade in rule_owned:
            b_bucket = before_by_after.get(grade)
            a_bucket = after_by_after.get(grade)
            if not b_bucket and not a_bucket:
                continue
            out.append(f"  by_after[{grade}]")
            for key in _bucket_keys(b_bucket, a_bucket):
                if key in {"n_helpful", "n_harmful", "n_neutral"}:
                    bv = (b_bucket or {}).get(key)
                    av = (a_bucket or {}).get(key)
                    out.append(
                        f"    {key:18s} "
                        f"{bv if bv is not None else '-'} -> "
                        f"{av if av is not None else '-'}"
                    )
                elif key == "override_precision":
                    bp = (b_bucket or {}).get(key)
                    ap = (a_bucket or {}).get(key)
                    out.append(f"    {key:18s} {_fmt_delta(bp, ap)}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "before",
        type=Path,
        help="Prior rule_engine_baseline.json (e.g. git show HEAD~1:...).",
    )
    parser.add_argument(
        "after",
        type=Path,
        help="Current rule_engine_baseline.json.",
    )
    parser.add_argument(
        "--rule-owned",
        nargs="+",
        default=["Poor", "Generic"],
        help="Grades to diff slice metrics and by_after buckets for.",
    )
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    print(diff_baselines(before, after, tuple(args.rule_owned)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
