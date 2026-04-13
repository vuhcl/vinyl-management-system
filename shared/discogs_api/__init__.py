"""Shared Discogs API client (import as ``shared.discogs_api``)."""

from shared.discogs_api.client import (
    DiscogsClient,
    discogs_client_from_env,
    get_user_collection,
    get_user_wantlist,
    personal_access_token_from_env,
)
from shared.discogs_api.oauth1 import (
    oauth_access_credentials_from_env,
    run_interactive_oauth,
)

__all__ = [
    "DiscogsClient",
    "discogs_client_from_env",
    "get_user_collection",
    "get_user_wantlist",
    "oauth_access_credentials_from_env",
    "personal_access_token_from_env",
    "run_interactive_oauth",
]

