"""
Discogs API client – shared by all subprojects.

Uses Discogs REST API with a personal access token (Settings → Developers →
Generate token). Env resolution: ``DISCOGS_USER_TOKEN``, then ``DISCOGS_TOKEN``
(same value many teams store under the shorter name, e.g. grader).
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

import pandas as pd
import requests

from shared.discogs_api.oauth1 import (
    make_oauth1_auth,
    oauth_access_credentials_from_env,
)
from shared.project_env import load_project_dotenv


def personal_access_token_from_env() -> str | None:
    """
    Personal token for ``Authorization: Discogs token=…``.

    Checks ``DISCOGS_USER_TOKEN`` first, then ``DISCOGS_TOKEN``.
    """
    for key in ("DISCOGS_USER_TOKEN", "DISCOGS_TOKEN"):
        raw = os.environ.get(key)
        if raw and str(raw).strip():
            return str(raw).strip()
    return None


def discogs_client_from_env() -> DiscogsClient | None:
    """
    Build a client from environment after loading repo-root ``.env``.

    Uses personal token (``DISCOGS_USER_TOKEN`` / ``DISCOGS_TOKEN``) if set;
    otherwise OAuth 1.0a access token + consumer key/secret from env.
    Returns ``None`` if no credentials are configured.
    """
    load_project_dotenv()
    client = DiscogsClient()
    if client.is_authenticated:
        return client
    return None


class DiscogsClient:
    """
    Thin client for Discogs API. All subprojects use this.

    Pass ``user_token`` explicitly, or leave it ``None`` to load from env
    (``DISCOGS_USER_TOKEN`` / ``DISCOGS_TOKEN``).
    """

    BASE = "https://api.discogs.com"

    def __init__(
        self,
        user_token: str | None = None,
        user_agent: str = "VinylManagementSystem/1.0",
        *,
        oauth: tuple[str, str, str, str] | None = None,
    ):
        self._oauth1: Any | None = None
        self.user_token: str | None = None

        if user_token is not None:
            self.user_token = str(user_token).strip() or None
        elif oauth is not None:
            ck, cs, at, ats = oauth
            if not (ck and cs and at and ats):
                raise ValueError("oauth= requires four non-empty strings")
            self._oauth1 = make_oauth1_auth(ck, cs, at, ats)
        else:
            self.user_token = personal_access_token_from_env()
            if not self.user_token:
                oc = oauth_access_credentials_from_env()
                if oc:
                    ck, cs, at, ats = oc
                    self._oauth1 = make_oauth1_auth(ck, cs, at, ats)

        self.user_agent = user_agent
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        if self.user_token:
            self._session.headers["Authorization"] = (
                f"Discogs token={self.user_token}"
            )

    @property
    def is_authenticated(self) -> bool:
        return bool(self.user_token) or self._oauth1 is not None

    def _params_for_get(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """
        Merge query params. For personal-token auth, Discogs sometimes requires
        ``token`` as a query parameter in addition to the Authorization header
        (see forum / client implementations); OAuth must not set this.
        """
        p = dict(params or {})
        if self.user_token:
            p["token"] = self.user_token
        return p

    def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        url = f"{self.BASE}{path}"
        kw: dict[str, Any] = {"params": self._params_for_get(params), "timeout": 30}
        if self._oauth1 is not None:
            kw["auth"] = self._oauth1
        r = self._session.get(url, **kw)
        r.raise_for_status()
        return r.json()

    def get_with_retries(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 45.0,
        max_retries: int = 8,
        backoff_base: float = 1.5,
        backoff_max: float = 120.0,
    ) -> dict[str, Any] | list[Any]:
        """
        GET JSON with retries on 429 (honors Retry-After), 5xx, and
        connection/timeout errors. Exponential backoff with jitter.
        """
        url = f"{self.BASE}{path}"
        kw: dict[str, Any] = {
            "params": self._params_for_get(params),
            "timeout": timeout,
        }
        if self._oauth1 is not None:
            kw["auth"] = self._oauth1

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                r = self._session.get(url, **kw)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After", "").strip()
                    if ra.isdigit():
                        wait = float(ra)
                    else:
                        wait = min(
                            backoff_max,
                            backoff_base * (2**attempt),
                        )
                    wait += random.uniform(0, min(5.0, 0.25 * wait))
                    time.sleep(wait)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    wait = min(backoff_max, backoff_base * (2**attempt))
                    wait += random.uniform(0, 1.0)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                try:
                    return r.json()
                except ValueError:
                    wait = min(backoff_max, backoff_base * (2**attempt))
                    time.sleep(wait + random.uniform(0, 1.0))
                    continue
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                wait = min(backoff_max, backoff_base * (2**attempt))
                time.sleep(wait + random.uniform(0, 1.0))
            except requests.HTTPError as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                if e.response is not None and e.response.status_code in (
                    500,
                    502,
                    503,
                    504,
                ):
                    wait = min(backoff_max, backoff_base * (2**attempt))
                    time.sleep(wait + random.uniform(0, 1.0))
                    continue
                raise

        if last_err:
            raise last_err
        raise RuntimeError("get_with_retries: exhausted without response")

    def _paginate(
        self, path: str, key: str = "releases", per_page: int = 100
    ) -> list[Any]:
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
        Returns list of dicts with 'id' (release_id),
        optionally 'basic_information'.
        """

        path = f"/users/{username}/collection/folders/{folder_id}/releases"
        return self._paginate(path, "releases")

    def get_user_wantlist(self, username: str) -> list[dict[str, Any]]:
        """Fetch user's wantlist.

        Returns list of dicts with 'id' (release_id).
        """
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

    def get_release(self, release_id: str | int) -> dict[str, Any]:
        """GET /releases/{release_id} — full release resource (community want/have, formats, etc.)."""
        rid = str(release_id).strip()
        return self._get(f"/releases/{rid}")

    def get_master(self, master_id: str | int) -> dict[str, Any]:
        """GET /masters/{master_id} — used for main_release vs release id (original pressing)."""
        mid = str(master_id).strip()
        return self._get(f"/masters/{mid}")

    def get_marketplace_stats(self, release_id: str | int) -> dict[str, Any]:
        """
        GET /marketplace/stats/{release_id}

        Returns keys commonly present: lowest_price, highest_price, num_for_sale,
        blocked_from_sale (bool). Discogs may include ``median`` or similar fields
        depending on API version — callers should normalize.
        """
        rid = str(release_id).strip()
        return self._get(f"/marketplace/stats/{rid}")

    def get_marketplace_stats_with_retries(
        self,
        release_id: str | int,
        *,
        max_retries: int = 8,
        backoff_base: float = 1.5,
        backoff_max: float = 120.0,
        timeout: float = 45.0,
    ) -> dict[str, Any]:
        """GET /marketplace/stats/{id} with rate-limit and transient error retries."""
        rid = str(release_id).strip()
        out = self.get_with_retries(
            f"/marketplace/stats/{rid}",
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
        )
        if not isinstance(out, dict):
            return {}
        return out

    def get_release_stats_with_retries(
        self,
        release_id: str | int,
        *,
        max_retries: int = 8,
        backoff_base: float = 1.5,
        backoff_max: float = 120.0,
        timeout: float = 45.0,
    ) -> dict[str, Any]:
        """
        GET /releases/{release_id}/stats — community ``num_want``, ``num_have``.

        Lightweight alternative to fetching the full release resource when only
        community counts matter (see Marketplace Statistics for listing floor).
        """
        rid = str(release_id).strip()
        out = self.get_with_retries(
            f"/releases/{rid}/stats",
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
        )
        if not isinstance(out, dict):
            return {}
        return out

    def get_release_with_retries(
        self,
        release_id: str | int,
        *,
        curr_abbr: str | None = None,
        max_retries: int = 8,
        backoff_base: float = 1.5,
        backoff_max: float = 120.0,
        timeout: float = 45.0,
    ) -> dict[str, Any]:
        """GET /releases/{id} with retries (community have/want, lowest_price, num_for_sale, …)."""
        rid = str(release_id).strip()
        params: dict[str, Any] = {}
        if curr_abbr:
            params["curr_abbr"] = curr_abbr
        out = self.get_with_retries(
            f"/releases/{rid}",
            params=params or None,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
        )
        if not isinstance(out, dict):
            return {}
        return out

    def get_price_suggestions_with_retries(
        self,
        release_id: str | int,
        *,
        max_retries: int = 8,
        backoff_base: float = 1.5,
        backoff_max: float = 120.0,
        timeout: float = 45.0,
    ) -> dict[str, Any]:
        """
        GET /marketplace/price_suggestions/{id} — grade → {value, currency}.

        Auth required; Discogs returns ``{}`` if seller settings are incomplete.
        """
        rid = str(release_id).strip()
        out = self.get_with_retries(
            f"/marketplace/price_suggestions/{rid}",
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
        )
        if not isinstance(out, dict):
            return {}
        return out

    def database_search(
        self,
        *,
        query: str = "",
        result_type: str = "release",
        page: int = 1,
        per_page: int = 100,
        sort: str | None = None,
        sort_order: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """
        GET /database/search (authenticated).

        Optional ``sort`` / ``sort_order`` match the website when Discogs
        supports them (e.g. ``sort=have``, ``sort_order=desc``); not all
        combinations are documented in the public API.
        """
        params: dict[str, Any] = {
            "q": query,
            "type": result_type,
            "page": max(1, page),
            "per_page": max(1, min(int(per_page), 100)),
        }
        if sort:
            params["sort"] = sort
        if sort_order:
            params["sort_order"] = sort_order
        for k, v in extra.items():
            if v is not None and k not in params:
                params[k] = v
        out = self.get_with_retries("/database/search", params=params)
        if not isinstance(out, dict):
            return {}
        return out


def get_user_collection(
    username: str,
    user_token: str | None = None,
    folder_id: int = 0,
) -> pd.DataFrame:
    """Fetch one user's collection as DataFrame.

    Token from arg or env (``DISCOGS_USER_TOKEN`` / ``DISCOGS_TOKEN``).
    """
    token = user_token or personal_access_token_from_env()
    if not token:
        return pd.DataFrame(columns=["user_id", "album_id"])
    client = DiscogsClient(user_token=token)
    return client.collection_to_dataframe(username, folder_id=folder_id)


def get_user_wantlist(
    username: str, user_token: str | None = None
) -> pd.DataFrame:
    """Fetch one user's wantlist as DataFrame.

    Token from arg or env.
    """
    token = user_token or personal_access_token_from_env()
    if not token:
        return pd.DataFrame(columns=["user_id", "album_id"])
    client = DiscogsClient(user_token=token)
    return client.wantlist_to_dataframe(username)
