"""
Discogs API client – shared by all subprojects.
Uses Discogs REST API with a user token (Settings → Developers → Generate token).
"""
from typing import Any

import pandas as pd
import requests


class DiscogsClient:
    """
    Thin client for Discogs API. All subprojects use this.
    Set DISCOGS_USER_TOKEN in env or pass user_token.
    """

    BASE = "https://api.discogs.com"

    def __init__(
        self,
        user_token: str | None = None,
        user_agent: str = "VinylManagementSystem/1.0",
    ):
        self.user_token = user_token
        self.user_agent = user_agent
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        if user_token:
            self._session.headers["Authorization"] = f"Discogs token={user_token}"

    def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        url = f"{self.BASE}{path}"
        r = self._session.get(url, params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def _paginate(self, path: str, key: str = "releases", per_page: int = 100) -> list[Any]:
        out: list[Any] = []
        page = 1
        while True:
            data = self._get(path, {"page": page, "per_page": per_page})
            if isinstance(data, dict):
                items = data.get(key, data.get("wants", []))
            else:
                items = data
            if not items:
                break
            out.extend(items)
            if len(items) < per_page:
                break
            page += 1
        return out

    def get_username(self) -> str:
        """Return authenticated user's username (requires token)."""
        data = self._get("/oauth/identity")
        return str(data.get("username", ""))

    def get_user_collection_releases(
        self, username: str, folder_id: int = 0
    ) -> list[dict[str, Any]]:
        """
        Fetch all releases in a user's collection folder.
        folder_id=0 is typically the main collection.
        Returns list of dicts with 'id' (release_id), optionally 'basic_information'.
        """
        path = f"/users/{username}/collection/folders/{folder_id}/releases"
        return self._paginate(path, "releases")

    def get_user_wantlist(self, username: str) -> list[dict[str, Any]]:
        """Fetch user's wantlist. Returns list of dicts with 'id' (release_id)."""
        path = f"/users/{username}/wants"
        return self._paginate(path, "wants")

    def collection_to_dataframe(
        self, username: str, folder_id: int = 0
    ) -> pd.DataFrame:
        """Collection as DataFrame: user_id, album_id (release_id)."""
        releases = self.get_user_collection_releases(username, folder_id)
        rows = []
        for r in releases:
            rid = r.get("id") or (r.get("basic_information") or {}).get("id")
            if rid is not None:
                rows.append({"user_id": username, "album_id": str(rid)})
        if not rows:
            return pd.DataFrame(columns=["user_id", "album_id"])
        return pd.DataFrame(rows, columns=["user_id", "album_id"])

    def wantlist_to_dataframe(self, username: str) -> pd.DataFrame:
        """Wantlist as DataFrame: user_id, album_id."""
        wants = self.get_user_wantlist(username)
        rows = [
            {"user_id": username, "album_id": str(w.get("id", ""))}
            for w in wants
            if w.get("id") is not None
        ]
        if not rows:
            return pd.DataFrame(columns=["user_id", "album_id"])
        return pd.DataFrame(rows, columns=["user_id", "album_id"])


def get_user_collection(
    username: str,
    user_token: str | None = None,
    folder_id: int = 0,
) -> pd.DataFrame:
    """Fetch one user's collection as DataFrame. Token from arg or DISCOGS_USER_TOKEN."""
    import os
    token = user_token or os.environ.get("DISCOGS_USER_TOKEN")
    if not token:
        return pd.DataFrame(columns=["user_id", "album_id"])
    client = DiscogsClient(user_token=token)
    return client.collection_to_dataframe(username, folder_id=folder_id)


def get_user_wantlist(
    username: str, user_token: str | None = None
) -> pd.DataFrame:
    """Fetch one user's wantlist as DataFrame. Token from arg or env."""
    import os
    token = user_token or os.environ.get("DISCOGS_USER_TOKEN")
    if not token:
        return pd.DataFrame(columns=["user_id", "album_id"])
    client = DiscogsClient(user_token=token)
    return client.wantlist_to_dataframe(username)
