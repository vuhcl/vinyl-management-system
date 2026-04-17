"""Ensemble blend sweep parsing from ``vinyliq.ensemble``."""
from __future__ import annotations

import pytest

from price_estimator.src.training.train_vinyliq import ensemble_blend_config_from_vinyliq


def test_blend_sweep_disabled_returns_no_pairs() -> None:
    v = {
        "ensemble": {
            "enabled": True,
            "blend": {"t": 4.0, "s": 0.35},
        }
    }
    cfg = ensemble_blend_config_from_vinyliq(v)
    assert cfg is not None
    assert cfg.get("blend_sweep_pairs") is None


def test_blend_sweep_cartesian() -> None:
    v = {
        "ensemble": {
            "enabled": True,
            "blend": {"t": 4.0, "s": 0.35},
            "blend_sweep": {"enabled": True, "t": [3.0, 4.0], "s": [0.2, 0.4]},
        }
    }
    cfg = ensemble_blend_config_from_vinyliq(v)
    assert cfg is not None
    pairs = cfg["blend_sweep_pairs"]
    assert pairs is not None
    assert len(pairs) == 4
    assert (3.0, 0.2) in pairs
    assert (4.0, 0.4) in pairs


def test_blend_sweep_explicit_pairs() -> None:
    v = {
        "ensemble": {
            "enabled": True,
            "blend": {"t": 4.0, "s": 0.35},
            "blend_sweep": {"enabled": True, "pairs": [[3.5, 0.3], [4.0, 0.35]]},
        }
    }
    cfg = ensemble_blend_config_from_vinyliq(v)
    assert cfg["blend_sweep_pairs"] == [(3.5, 0.3), (4.0, 0.35)]


def test_blend_sweep_empty_lists_fallback() -> None:
    v = {
        "ensemble": {
            "enabled": True,
            "blend": {"t": 2.5, "s": 0.1},
            "blend_sweep": {"enabled": True, "t": [], "s": [0.2]},
        }
    }
    cfg = ensemble_blend_config_from_vinyliq(v)
    assert cfg["blend_sweep_pairs"] == [(2.5, 0.1)]


def test_blend_sweep_invalid_pairs_raises() -> None:
    v = {
        "ensemble": {
            "enabled": True,
            "blend": {"t": 4.0, "s": 0.35},
            "blend_sweep": {"enabled": True, "pairs": []},
        }
    }
    with pytest.raises(ValueError, match="empty"):
        ensemble_blend_config_from_vinyliq(v)
