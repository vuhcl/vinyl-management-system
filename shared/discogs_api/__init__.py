"""Shared Discogs API client (import as ``shared.discogs_api``)."""

from shared.discogs_api.client import (
    DiscogsClient,
    get_user_collection,
    get_user_wantlist,
)

__all__ = ["DiscogsClient", "get_user_collection", "get_user_wantlist"]

