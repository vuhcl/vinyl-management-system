"""
Discogs authentication for the web app.

Supports:
- Token-based: user provides Discogs personal token (Settings → Developers).
- Future: OAuth 1.0a flow for "Login with Discogs" (store token in session).
"""
from __future__ import annotations

import os
from typing import Any

# In-memory store for development; replace with Redis/DB session in production.
_user_tokens: dict[str, str] = {}


def set_user_token(username: str, token: str) -> None:
    """Store Discogs token for a user (e.g. after OAuth or token paste)."""
    _user_tokens[username] = token


def get_user_token(username: str) -> str | None:
    """Return stored token for user, or None."""
    return _user_tokens.get(username)


def get_token_for_request(
    username: str | None,
    request_env: dict[str, Any] | None = None,
) -> str | None:
    """
    Resolve Discogs token: stored per-user first, then env personal token.

    Env: ``DISCOGS_USER_TOKEN``, then ``DISCOGS_TOKEN``.
    request_env can hold session (e.g. from FastAPI request.state).
    """
    if username and get_user_token(username):
        return get_user_token(username)
    for key in ("DISCOGS_USER_TOKEN", "DISCOGS_TOKEN"):
        t = os.environ.get(key)
        if t and str(t).strip():
            return str(t).strip()
    return None
