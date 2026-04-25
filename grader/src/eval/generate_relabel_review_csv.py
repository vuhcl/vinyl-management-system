"""
Build a skim-friendly CSV of Cleanlab media candidates + suggested labels.

Cohort is the last ``--patches-tail`` JSONL lines in ``label_patches_path``
(default 287). Each patch is joined to ``--cleanlab-media`` on
``(source, item_id)`` to recover **pre-patch** grades and ``oof_pred_label``
from the audit snapshot, then compared to the patch's new grades.

**suggestion_source_media** values:

- ``cohort_media_old_oof``: plurality of new_media among relabels with the
  same ``(old_media, oof_pred)`` pair; requires **≥ 2** cohort rows.
- ``cohort_media_old_only``: plurality of new_media among relabels with the
  same ``old_media`` only (any OOF); requires **≥ 3** cohort rows.
- ``oof_pred_fallback``: use the row's ``oof_pred_label`` as suggested media.

**Suggested sleeve**: plurality of new_sleeve among relabels with the same
``old_sleeve`` (from audit), if **≥ 2** such rows; else **empty string**.

Usage (from repo root)::

    uv run python -m grader.src.eval.generate_relabel_review_csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_DEFAULTS = {
    "patches": Path("grader/data/label_patches.jsonl"),
    "cleanlab_media": Path("grader/reports/cleanlab_label_audit_media.csv"),
    "candidates": Path(
        "grader/reports/"
        "cleanlab_media_candidates_like_relabels_high_priority_text_enriched.csv"
    ),
    "candidates_wide": Path(
        "grader/reports/cleanlab_media_candidates_like_relabels.csv"
    ),
    "output": Path("grader/reports/cleanlab_media_review_suggested_labels.csv"),
}


def _row_key(item_id: str, source: str) -> tuple[str, str]:
    return (str(source).strip(), str(item_id).strip())


def load_jsonl_tail(path: Path, n: int) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = [ln for ln in lines[-n:] if ln.strip() and not ln.strip().startswith("#")]
    out: list[dict[str, Any]] = []
    for ln in tail:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            logger.warning("skip invalid JSON in %s: %s", path, e)
    return out


def load_cleanlab_media_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    idx: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            k = _row_key(str(row.get("item_id", "")), str(row.get("source", "")))
            idx[k] = {str(a): (str(b) if b is not None else "") for a, b in row.items()}
    return idx


def build_cohort_stats(
    patches: list[dict[str, Any]],
    cl_index: dict[tuple[str, str], dict[str, str]],
) -> tuple[
    defaultdict[tuple[str, str], Counter[str]],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
]:
    """
    Returns:
        pair_new: (old_media, oof_pred) -> Counter[new_media]
        marginal: (old_media, new_media) -> count
        sleeve_marginal: (old_sleeve, new_sleeve) -> count
    """
    pair_new: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    marginal: Counter[tuple[str, str]] = Counter()
    sleeve_marginal: Counter[tuple[str, str]] = Counter()

    for p in patches:
        sid = str(p.get("source", "")).strip()
        iid = str(p.get("item_id", "")).strip()
        if not sid or not iid:
            continue
        new_m = p.get("media_label")
        if new_m is None:
            continue
        new_m = str(new_m).strip()
        if not new_m:
            continue
        cr = cl_index.get((sid, iid))
        if not cr:
            continue
        old_m = str(cr.get("media_label", "")).strip()
        oof = str(cr.get("oof_pred_label", "")).strip()
        pair_new[(old_m, oof)][new_m] += 1
        marginal[(old_m, new_m)] += 1

        new_s = p.get("sleeve_label")
        if new_s is not None:
            new_s = str(new_s).strip()
            old_s = str(cr.get("sleeve_label", "")).strip()
            if new_s:
                sleeve_marginal[(old_s, new_s)] += 1

    return pair_new, marginal, sleeve_marginal


def suggest_media_label(
    old_media: str,
    oof_pred: str,
    pair_new: defaultdict[tuple[str, str], Counter[str]],
    marginal: Counter[tuple[str, str]],
) -> tuple[str, str]:
    ctr = pair_new.get((old_media, oof_pred), Counter())
    if sum(ctr.values()) >= 2:
        return ctr.most_common(1)[0][0], "cohort_media_old_oof"
    sub = Counter()
    for (m, nm), c in marginal.items():
        if m == old_media:
            sub[nm] += c
    if sum(sub.values()) >= 3:
        return sub.most_common(1)[0][0], "cohort_media_old_only"
    return oof_pred, "oof_pred_fallback"


def suggest_sleeve_label(
    old_sleeve: str,
    sleeve_marginal: Counter[tuple[str, str]],
) -> str:
    sub = Counter()
    for (os, ns), c in sleeve_marginal.items():
        if os == old_sleeve:
            sub[ns] += c
    if sum(sub.values()) >= 2:
        return sub.most_common(1)[0][0]
    return ""


def write_review_csv(
    *,
    patches_path: Path,
    patches_tail: int,
    cleanlab_media_path: Path,
    candidates_csv: Path,
    output_path: Path,
    snippet_max: int = 240,
) -> dict[str, int | str]:
    patches = load_jsonl_tail(patches_path, patches_tail)
    cl_index = load_cleanlab_media_index(cleanlab_media_path)
    pair_new, marginal, sleeve_marginal = build_cohort_stats(patches, cl_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    fieldnames = [
        "item_id",
        "source",
        "current_media_label",
        "current_sleeve_label",
        "suggested_media_label",
        "suggested_sleeve_label",
        "suggestion_source_media",
        "oof_pred_label",
        "cleanlab_self_confidence",
        "text_pattern_score",
        "text_token_hits",
        "modeling_text_snippet",
    ]
    with candidates_csv.open(encoding="utf-8", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        rdr = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for row in rdr:
            n_in += 1
            old_m = str(row.get("media_label_at_audit", "")).strip()
            oof = str(row.get("oof_pred_label", "")).strip()
            old_s = str(row.get("sleeve_label", "")).strip()
            sm, src_m = suggest_media_label(old_m, oof, pair_new, marginal)
            ss = suggest_sleeve_label(old_s, sleeve_marginal)
            snip = str(row.get("modeling_text_snippet") or "")
            if len(snip) > snippet_max:
                snip = snip[:snippet_max]
            w.writerow(
                {
                    "item_id": row.get("item_id", ""),
                    "source": row.get("source", ""),
                    "current_media_label": old_m,
                    "current_sleeve_label": old_s,
                    "suggested_media_label": sm,
                    "suggested_sleeve_label": ss,
                    "suggestion_source_media": src_m,
                    "oof_pred_label": oof,
                    "cleanlab_self_confidence": row.get(
                        "cleanlab_self_confidence", ""
                    ),
                    "text_pattern_score": row.get("text_pattern_score", ""),
                    "text_token_hits": row.get("text_token_hits", ""),
                    "modeling_text_snippet": snip,
                }
            )
            n_out += 1
    logger.info(
        "Wrote %s (%d rows from %d input lines).",
        output_path,
        n_out,
        n_in,
    )
    return {
        "patches_used": len(patches),
        "input_rows": n_in,
        "output_rows": n_out,
        "output": str(output_path),
    }


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Generate review CSV with suggested media/sleeve labels."
    )
    p.add_argument(
        "--patches-path",
        type=Path,
        default=REPO_DEFAULTS["patches"],
    )
    p.add_argument(
        "--patches-tail",
        type=int,
        default=287,
        help="Number of trailing JSONL lines to use as the relabel cohort.",
    )
    p.add_argument(
        "--cleanlab-media",
        type=Path,
        default=REPO_DEFAULTS["cleanlab_media"],
    )
    p.add_argument(
        "--candidates-csv",
        type=Path,
        default=REPO_DEFAULTS["candidates"],
        help="High-priority enriched candidates CSV (default: HP text-enriched).",
    )
    p.add_argument(
        "--use-wide-candidates",
        type=Path,
        nargs="?",
        const=REPO_DEFAULTS["candidates_wide"],
        default=None,
        help=(
            "Use wider ~7k candidate list; optional path overrides default wide file."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_DEFAULTS["output"],
    )
    p.add_argument(
        "--snippet-max",
        type=int,
        default=240,
    )
    args = p.parse_args(argv)
    candidates = (
        args.use_wide_candidates
        if args.use_wide_candidates is not None
        else args.candidates_csv
    )
    for path, label in (
        (args.patches_path, "patches"),
        (args.cleanlab_media, "cleanlab media"),
        (candidates, "candidates"),
    ):
        if not path.is_file():
            print(f"Missing {label} file: {path}", file=sys.stderr)
            return 1
    stats = write_review_csv(
        patches_path=args.patches_path,
        patches_tail=args.patches_tail,
        cleanlab_media_path=args.cleanlab_media,
        candidates_csv=candidates,
        output_path=args.output,
        snippet_max=args.snippet_max,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
