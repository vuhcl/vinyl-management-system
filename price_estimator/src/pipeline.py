"""
Price estimator pipeline (stub).

End-to-end: ingest price data → features → train → save artifacts.
Predict: release_id [+ condition] → estimated price(s).
"""
from pathlib import Path


def load_config(config_path: Path | None = None) -> dict:
    from core.config import get_project_root, load_config as _load
    root = get_project_root()
    path = config_path or root / "price_estimator" / "configs" / "base.yaml"
    return _load(path)


def run_pipeline(
    config_path: Path | None = None,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    skip_ingest: bool = False,
) -> dict:
    """
    Run training pipeline. Stub: no model training yet; creates artifact dir.
    """
    from core.config import get_project_root
    cfg = load_config(config_path)
    root = get_project_root()
    paths = cfg.get("paths") or {}
    data_dir = Path(data_dir or paths.get("raw_data", root / "data" / "raw"))
    artifacts_dir = Path(artifacts_dir or paths.get("artifacts", root / "artifacts" / "price_estimator"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Stub: no ingest/train yet
    return {"artifacts_dir": artifacts_dir, "config": cfg}


def estimate(
    release_id: str,
    sleeve_condition: str | None = None,
    media_condition: str | None = None,
    artifacts_dir: Path | None = None,
) -> dict:
    """
    Return price estimate for a release. Stub: returns placeholder.
    """
    return {
        "release_id": release_id,
        "sleeve_condition": sleeve_condition,
        "media_condition": media_condition,
        "estimate_usd": None,
        "interval_low": None,
        "interval_high": None,
        "status": "stub",
    }
