"""Shared FastAPI dependencies and request helpers."""

from fastapi import Request


def get_current_username(request: Request) -> str | None:
    """Username from request state (set by session middleware from cookie)."""
    return getattr(request.state, "username", None)
