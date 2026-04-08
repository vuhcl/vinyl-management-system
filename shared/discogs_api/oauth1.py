"""
Discogs OAuth 1.0a helpers (PLAINTEXT signatures).

Discogs documents: request token → authorize in browser → access token.
URLs default to production; override with env if Discogs changes endpoints.
"""
from __future__ import annotations

import os
from typing import Any


def oauth_urls_from_env() -> dict[str, str]:
    return {
        "request_token": os.environ.get(
            "DISCOGS_REQUEST_TOKEN_URL",
            "https://api.discogs.com/oauth/request_token",
        ).strip(),
        "authorize": os.environ.get(
            "DISCOGS_AUTHORIZE_URL",
            "https://www.discogs.com/oauth/authorize",
        ).strip(),
        "access_token": os.environ.get(
            "DISCOGS_ACCESS_TOKEN_URL",
            "https://api.discogs.com/oauth/access_token",
        ).strip(),
    }


def oauth_access_credentials_from_env() -> tuple[str, str, str, str] | None:
    """
    Return (consumer_key, consumer_secret, access_token, access_token_secret)
    if all are set.
    """
    ck = (os.environ.get("DISCOGS_CONSUMER_KEY") or "").strip()
    cs = (os.environ.get("DISCOGS_CONSUMER_SECRET") or "").strip()
    at = (os.environ.get("DISCOGS_OAUTH_TOKEN") or "").strip()
    ats = (os.environ.get("DISCOGS_OAUTH_TOKEN_SECRET") or "").strip()
    if ck and cs and at and ats:
        return (ck, cs, at, ats)
    return None


def run_interactive_oauth(
    consumer_key: str,
    consumer_secret: str,
    *,
    user_agent: str = "VinylManagementSystem/1.0",
) -> tuple[str, str]:
    """
    Run the browser leg once; return (access_token, access_token_secret).

    Requires ``DISCOGS_OAUTH_CALLBACK`` in env (or *callback_uri*) to match the
    callback URL registered for your Discogs application.
    """
    from requests_oauthlib import OAuth1Session

    urls = oauth_urls_from_env()
    callback_uri = (os.environ.get("DISCOGS_OAUTH_CALLBACK") or "").strip()
    if not callback_uri:
        callback_uri = "http://127.0.0.1:8765/callback"
        print(
            "DISCOGS_OAUTH_CALLBACK not set; using "
            "http://127.0.0.1:8765/callback\n"
            "Register this callback URL on your Discogs app, or set "
            "DISCOGS_OAUTH_CALLBACK to the URL you registered.",
            flush=True,
        )

    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        callback_uri=callback_uri,
    )
    oauth.headers["User-Agent"] = user_agent
    oauth.fetch_request_token(urls["request_token"])
    authorization_url = oauth.authorization_url(urls["authorize"])
    print("\nOpen this URL in your browser and approve the app:\n")
    print(authorization_url, "\n", flush=True)
    print(
        "After approval, Discogs redirects to your callback with "
        "?oauth_verifier=... in the URL.\n"
        "Paste the **full redirect URL** or **only oauth_verifier**:\n",
        flush=True,
    )
    pasted = input().strip()
    if "oauth_verifier=" in pasted:
        oauth.parse_authorization_response(
            pasted if pasted.startswith("http") else f"https://dummy/?{pasted}"
        )
    else:
        token = oauth.fetch_access_token(
            urls["access_token"],
            verifier=pasted,
        )
        return str(token["oauth_token"]), str(token["oauth_token_secret"])

    token = oauth.fetch_access_token(urls["access_token"])
    return str(token["oauth_token"]), str(token["oauth_token_secret"])


def make_oauth1_auth(
    consumer_key: str,
    consumer_secret: str,
    access_token: str,
    access_token_secret: str,
) -> Any:
    """OAuth1 auth object for ``requests`` (PLAINTEXT, per Discogs docs)."""
    from requests_oauthlib import OAuth1

    return OAuth1(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret,
        signature_method="PLAINTEXT",
    )
