"""
Shared Discogs API client.

This mirrors the previous top-level `discogs_api` package so all subprojects
can import via `shared.discogs_api`.
"""

from shared.discogs_api.client import (
    DiscogsClient,
    get_user_collection,
    get_user_wantlist,
)

__all__ = ["DiscogsClient", "get_user_collection", "get_user_wantlist"]

