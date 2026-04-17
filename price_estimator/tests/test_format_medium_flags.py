"""Format / medium flags for VinylIQ catalog tail."""
from __future__ import annotations

import pytest

from price_estimator.src.features.vinyliq_features import format_medium_flags


def test_format_medium_flags_7_vs_lp():
    fmts = [{"name": "Vinyl", "descriptions": ['7"', "45 RPM"]}]
    m = format_medium_flags(fmts, None)
    assert m["is_7inch"] == pytest.approx(1.0)
    assert m["is_lp"] == pytest.approx(0.0)
    assert m["format_family"] == pytest.approx(2.0)


def test_format_medium_flags_box_set_priority():
    fmts = [{"name": "Vinyl", "descriptions": ["LP", "Box Set"], "qty": "3"}]
    m = format_medium_flags(fmts, None)
    assert m["is_box_set"] == pytest.approx(1.0)
    assert m["is_multi_disc"] == pytest.approx(0.0)
    assert m["format_family"] == pytest.approx(5.0)


def test_format_medium_flags_10inch():
    m = format_medium_flags([], '10", Vinyl')
    assert m["is_10inch"] == pytest.approx(1.0)
    assert m["format_family"] == pytest.approx(3.0)


def test_format_medium_flags_qty_multi():
    fmts = [{"name": "Vinyl", "descriptions": ["LP"], "qty": "2"}]
    m = format_medium_flags(fmts, None)
    assert m["is_multi_disc"] == pytest.approx(1.0)
    assert m["is_box_set"] == pytest.approx(0.0)
