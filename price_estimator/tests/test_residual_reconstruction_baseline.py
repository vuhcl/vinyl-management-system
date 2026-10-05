"""R0: Baseline for residual / log1p-dollar reconstruction (R8 regression lock).

Regenerate fixture (runs before tests via conftest):
  VINYLIQ_REGEN_RESIDUAL_BASELINE=1 uv run pytest \
    price_estimator/tests/test_residual_reconstruction_baseline.py -q

Comparisons use float closeness (not exact repr strings): libm / platform ULP
drift is ~1e-15 and would flake CI vs Mac fixture hosts.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from price_estimator.src.models.fitted_regressor import (
    TARGET_KIND_DOLLAR_LOG1P,
    TARGET_KIND_RESIDUAL_LOG_MEDIAN,
    pred_log1p_dollar_for_metrics,
)
from price_estimator.tests.residual_baseline_fixtures import (
    FIXTURE_PATH,
    _pyfunc_prices,
    generate_baseline_dict,
)

_FIXTURE = FIXTURE_PATH
# ponytail: ~few ULPs across platforms; tighten if a real regression slips through
_RTOL = 1e-12
_ATOL = 1e-12


def _assert_close(got: float, exp: float, *, where: str) -> None:
    assert math.isclose(got, exp, rel_tol=_RTOL, abs_tol=_ATOL), (
        f"{where}: {got!r} !~ {exp!r} (rtol={_RTOL}, atol={_ATOL})"
    )


def _assert_repr_list_close(got: list[str], exp: list[str], *, where: str) -> None:
    assert len(got) == len(exp), f"{where}: len {len(got)} != {len(exp)}"
    for i, (g, e) in enumerate(zip(got, exp)):
        _assert_close(float(g), float(e), where=f"{where}[{i}]")


@pytest.fixture(scope="module")
def baseline_fixture():
    if not _FIXTURE.is_file():
        pytest.fail(
            f"Missing {_FIXTURE}; run with "
            "VINYLIQ_REGEN_RESIDUAL_BASELINE=1 to generate."
        )
    return json.loads(_FIXTURE.read_text())


def test_fixture_file_present_when_not_regen() -> None:
    if os.environ.get("VINYLIQ_REGEN_RESIDUAL_BASELINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip()
    assert _FIXTURE.is_file(), "Run with VINYLIQ_REGEN_RESIDUAL_BASELINE=1 once"


def test_pred_log1p_dollar_grid_matches_baseline(baseline_fixture: dict) -> None:
    expected = {
        (float(r["pred"]), float(r["med"]), str(r["kind"])): float(r["out"])
        for r in baseline_fixture["pred_log1p_dollar_for_metrics"]
    }
    preds = [-2.0, -0.5, 0.0, 0.5, 2.0]
    meds = [0.0, 1.0, 5.0, 25.0, 100.0, 1000.0]
    kinds = [TARGET_KIND_DOLLAR_LOG1P, TARGET_KIND_RESIDUAL_LOG_MEDIAN]
    for kind in kinds:
        for p in preds:
            for m in meds:
                lp = pred_log1p_dollar_for_metrics(
                    np.array([p], dtype=np.float64),
                    np.array([m], dtype=np.float64),
                    kind,
                )
                got = float(lp[0])
                exp = expected[(p, m, kind)]
                _assert_close(got, exp, where=f"pred={p} med={m} kind={kind}")


def test_pyfunc_single_residual_6row_matches_baseline(baseline_fixture: dict) -> None:
    got = _pyfunc_prices(
        ensemble=None, meds=[0.0, 1.0, 5.0, 25.0, 100.0, 1000.0]
    )
    _assert_repr_list_close(
        got, baseline_fixture["pyfunc_single_residual_6row"], where="single"
    )


def test_pyfunc_ensemble_residual_t4_s1_matches_baseline(baseline_fixture: dict) -> None:
    got = _pyfunc_prices(
        ensemble={
            "enabled": True,
            "blend": {"kind": "log_anchor_sigmoid", "t": 4.0, "s": 1.0},
            "regressor_nm": "regressor_ensemble_nm.joblib",
            "regressor_ord": "regressor_ensemble_ord.joblib",
        },
        meds=[0.0, 1.0, 5.0, 25.0, 100.0, 1000.0],
    )
    _assert_repr_list_close(
        got, baseline_fixture["pyfunc_ensemble_residual_t4_s1"], where="ens_t4_s1"
    )


def test_pyfunc_ensemble_residual_t2_5_s0_5_matches_baseline(baseline_fixture: dict) -> None:
    got = _pyfunc_prices(
        ensemble={
            "enabled": True,
            "blend": {"kind": "log_anchor_sigmoid", "t": 2.5, "s": 0.5},
            "regressor_nm": "regressor_ensemble_nm.joblib",
            "regressor_ord": "regressor_ensemble_ord.joblib",
        },
        meds=[0.0, 1.0, 5.0, 25.0, 100.0, 1000.0],
    )
    _assert_repr_list_close(
        got,
        baseline_fixture["pyfunc_ensemble_residual_t2.5_s0.5"],
        where="ens_t2.5_s0.5",
    )


def test_generate_matches_loaded_fixture() -> None:
    """Sanity: in-memory generation matches on-disk JSON within float tol."""
    if not _FIXTURE.is_file():
        pytest.skip("No fixture yet")
    disk = json.loads(_FIXTURE.read_text())
    fresh = generate_baseline_dict()
    assert disk["version"] == fresh["version"]
    assert len(disk["pred_log1p_dollar_for_metrics"]) == len(
        fresh["pred_log1p_dollar_for_metrics"]
    )
    for i, (d, f) in enumerate(
        zip(disk["pred_log1p_dollar_for_metrics"], fresh["pred_log1p_dollar_for_metrics"])
    ):
        assert d["pred"] == f["pred"] and d["med"] == f["med"] and d["kind"] == f["kind"]
        _assert_close(float(d["out"]), float(f["out"]), where=f"grid[{i}]")
    for key in (
        "pyfunc_single_residual_6row",
        "pyfunc_ensemble_residual_t4_s1",
        "pyfunc_ensemble_residual_t2.5_s0.5",
    ):
        _assert_repr_list_close(fresh[key], disk[key], where=key)
