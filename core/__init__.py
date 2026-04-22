"""
Core shared layer for the Vinyl Management System.

- config: Load and merge YAML configs (base + component overrides).
- auth: Discogs token/session handling for the web app.
- jobs: Ingest jobs (Discogs collection/wantlist → data/raw at repo root or per-user).
"""
from core.config import load_config, get_project_root

__all__ = ["load_config", "get_project_root"]
