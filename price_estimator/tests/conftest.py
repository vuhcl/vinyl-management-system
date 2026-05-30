"""Pytest hooks for price_estimator tests."""
from __future__ import annotations

import os
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Skip monitoring modules when optional deps are not installed.

    Tier A CI uses ``uv sync --extra test`` only. ``-m "not monitoring"`` does not
    prevent collection-time imports of ``test_monitoring_*.py`` (statsmodels via
    ``drift_stats``). Tier D workflows install ``--extra monitoring`` first.
    """
    if not collection_path.name.startswith("test_monitoring"):
        return None
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        return True
    return None


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
