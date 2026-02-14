"""
Load and merge configuration for the master project and all ML components.

Config precedence: base.yaml (paths, discogs, aoty) + optional component overrides.
All paths in config are resolved relative to project root unless absolute.
"""
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Project root = directory containing configs/ and pyproject.toml."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "configs").exists() or (root / "pyproject.toml").exists(), f"Invalid project root: {root}"
    return root


def load_config(
    config_path: Path | str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load base config from configs/base.yaml (or given path) and optionally merge overrides.

    Relative paths in config (paths.raw_data, paths.processed_data, aoty_scraped.dir)
    are resolved against project root.
    """
    root = get_project_root()
    path = Path(config_path) if config_path else root / "configs" / "base.yaml"
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return dict(overrides or {})

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        _deep_merge(cfg, overrides)

    # Resolve relative paths under paths.* and aoty_scraped.dir
    paths_cfg = cfg.get("paths") or {}
    for key in ("raw_data", "processed_data", "artifacts"):
        if key in paths_cfg and paths_cfg[key]:
            p = Path(paths_cfg[key])
            if not p.is_absolute():
                paths_cfg[key] = str(root / p)
    aoty = cfg.get("aoty_scraped") or {}
    if aoty.get("dir"):
        d = Path(aoty["dir"])
        if not d.is_absolute():
            aoty["dir"] = str(root / d)
        cfg["aoty_scraped"] = aoty

    return cfg


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
