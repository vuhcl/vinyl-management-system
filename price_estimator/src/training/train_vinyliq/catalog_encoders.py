"""Frequency-capped catalog id encoders and on-disk encoder artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ...models.condition_adjustment import default_params, save_params


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

def _write_encoder_artifacts(model_dir: Path, catalog_encoders: dict[str, dict[str, float]]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "catalog_encoders.json").write_text(json.dumps(catalog_encoders, indent=2))
    (model_dir / "genre_encoder.json").write_text(
        json.dumps(catalog_encoders.get("genre", {}), indent=2),
    )
    save_params(model_dir / "condition_params.json", default_params())
