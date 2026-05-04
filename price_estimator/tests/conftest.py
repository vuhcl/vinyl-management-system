"""Pytest hooks for price_estimator tests."""
from __future__ import annotations

import os


def pytest_sessionstart(session) -> None:
    if os.environ.get("VINYLIQ_REGEN_RESIDUAL_BASELINE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    from price_estimator.tests.residual_baseline_fixtures import write_baseline_fixture

    path = write_baseline_fixture()
    session.config.pluginmanager.getplugin("terminalreporter")
    print(f"\n[VINYLIQ] Wrote residual baseline fixture: {path}\n")
