"""Runtime YAML merge (inherits chain + inference condition overlays)."""

from __future__ import annotations

import pytest

from price_estimator.src.inference.service import (
    load_yaml_config,
    yaml_inference_condition_overlay,
)
from price_estimator.src.models.condition_adjustment import (
    merge_inference_condition_params,
)


def test_merge_inference_prefers_yaml_alpha_beta_over_artifact_defaults() -> None:
    artifact = {"alpha": 0.06, "beta": 0.04, "ref_grade": 8.0, "grade_delta_scale": {}}
    overlay = {"alpha": 0.17885, "beta": 0.13049, "ref_grade": 8.0}
    m = merge_inference_condition_params(artifact, overlay)
    assert m["alpha"] == pytest.approx(0.17885)
    assert m["beta"] == pytest.approx(0.13049)


def test_yaml_inference_overlay_extracts_ordinal_cascade() -> None:
    cfg = {
        "vinyliq": {
            "training_label": {
                "sale_floor_blend": {
                    "ordinal_cascade": {
                        "condition_adjustment": {"alpha": 0.15, "beta": 0.12, "ref_grade": 8.0},
                        "grade_delta_scale": {"price_ref_usd": 99.0, "price_gamma": 0.1},
                    }
                }
            }
        }
    }
    o = yaml_inference_condition_overlay(cfg)
    assert o is not None
    assert o["alpha"] == pytest.approx(0.15)
    merged = merge_inference_condition_params(
        {"alpha": 0.01, "beta": 0.01, "ref_grade": 8.0, "grade_delta_scale": {}}, o,
    )
    gd = merged.get("grade_delta_scale")
    assert isinstance(gd, dict)
    assert gd.get("price_ref_usd") == pytest.approx(99.0)


def test_load_yaml_config_inherits_merges_child_over_parent(tmp_path: Path) -> None:
    base = tmp_path / "configs"
    base.mkdir()
    (base / "base.yaml").write_text(
        """vinyliq:
  training_label:
    sale_floor_blend:
      ordinal_cascade:
        condition_adjustment:
          alpha: 0.1
          beta: 0.2
          ref_grade: 8.0
""",
        encoding="utf-8",
    )

    overlay_p = tmp_path / "kube.yaml"
    overlay_p.write_text(
        """inherits: configs/base.yaml
vinyliq:
  paths:
    postgres_dsn_env: DATABASE_URL
""",
        encoding="utf-8",
    )

    cfg = load_yaml_config(overlay_p, workspace_root=tmp_path)
    assert cfg["vinyliq"]["paths"]["postgres_dsn_env"] == "DATABASE_URL"

    coef = yaml_inference_condition_overlay(cfg)
    assert coef is not None
    assert coef["alpha"] == pytest.approx(0.1)


def test_cluster_overlay_inherits_vinyliq_base_preserves_inference(tmp_path):
    """GKE overlay must inherit ``price_estimator/configs/*``, not root ``configs/base``."""
    pe = tmp_path / "price_estimator" / "configs"
    pe.mkdir(parents=True)
    (pe / "base.yaml").write_text(
        """vinyliq:
  inference:
    use_price_suggestion_condition_anchor: true
  paths:
    model_dir: artifacts/vinyliq
""",
        encoding="utf-8",
    )
    kube = tmp_path / "kube.yaml"
    kube.write_text(
        """inherits: price_estimator/configs/base.yaml
vinyliq:
  paths:
    postgres_dsn_env: DATABASE_URL
    model_dir: /data/artifacts/vinyliq
""",
        encoding="utf-8",
    )

    merged = load_yaml_config(kube, workspace_root=tmp_path)
    v = merged.get("vinyliq") or {}
    inf = v.get("inference") or {}
    assert inf.get("use_price_suggestion_condition_anchor") is True
    assert v["paths"]["postgres_dsn_env"] == "DATABASE_URL"
    assert v["paths"]["model_dir"] == "/data/artifacts/vinyliq"
