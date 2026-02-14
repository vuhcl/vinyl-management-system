"""
Shared Discogs API client for all vinyl_management_system subprojects.
Use this package for any Discogs data (collection, wantlist, releases, etc.).
"""
from discogs_api.client import DiscogsClient, get_user_collection, get_user_wantlist

__all__ = ["DiscogsClient", "get_user_collection", "get_user_wantlist"]
