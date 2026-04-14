import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from enum import StrEnum
from typing import Any, Generic, TypedDict, TypeVar

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SearchType(StrEnum):
    """Item types accepted by the Spotify search endpoint."""

    TRACK = "track"
    ALBUM = "album"
    ARTIST = "artist"
    PLAYLIST = "playlist"
    SHOW = "show"
    EPISODE = "episode"
    AUDIOBOOK = "audiobook"


# ---------------------------------------------------------------------------
# TypedDicts — shared primitives
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


class ExternalUrls(TypedDict):
    spotify: str


class Image(TypedDict, total=False):
    url: str
    height: int | None
    width: int | None


class Followers(TypedDict, total=False):
    href: str | None
    total: int


class Restrictions(TypedDict, total=False):
    reason: str  # "market" | "product" | "explicit"


class ExternalIds(TypedDict, total=False):
    isrc: str
    ean: str
    upc: str


class CopyrightObject(TypedDict, total=False):
    text: str
    type: str


class AuthorObject(TypedDict):
    name: str


class NarratorObject(TypedDict):
    name: str


# ---------------------------------------------------------------------------
# TypedDicts — artist / album / track
# ---------------------------------------------------------------------------


class SimplifiedArtistObject(TypedDict, total=False):
    external_urls: ExternalUrls
    href: str
    id: str
    name: str
    type: str
    uri: str


class AlbumObject(TypedDict, total=False):
    album_type: str  # "album" | "single" | "compilation"
    total_tracks: int
    available_markets: list[str]
    external_urls: ExternalUrls
    href: str
    id: str
    images: list[Image]
    name: str
    release_date: str
    release_date_precision: str  # "year" | "month" | "day"
    restrictions: Restrictions
    type: str
    uri: str
    artists: list[SimplifiedArtistObject]


class TrackObject(TypedDict, total=False):
    album: AlbumObject
    artists: list[SimplifiedArtistObject]
    available_markets: list[str]
    disc_number: int
    duration_ms: int
    explicit: bool
    external_ids: ExternalIds
    external_urls: ExternalUrls
    href: str
    id: str
    is_playable: bool
    linked_from: dict[str, Any]
    restrictions: Restrictions
    name: str
    popularity: int
    preview_url: str | None
    track_number: int
    type: str
    uri: str
    is_local: bool


class ArtistObject(TypedDict, total=False):
    external_urls: ExternalUrls
    followers: Followers
    genres: list[str]
    href: str
    id: str
    images: list[Image]
    name: str
    popularity: int
    type: str
    uri: str


# ---------------------------------------------------------------------------
# TypedDicts — playlist
# ---------------------------------------------------------------------------


class PlaylistOwner(TypedDict, total=False):
    external_urls: ExternalUrls
    href: str
    id: str
    type: str
    uri: str
    display_name: str | None


class PlaylistTracksRef(TypedDict, total=False):
    href: str
    total: int


class SimplifiedPlaylistObject(TypedDict, total=False):
    collaborative: bool
    description: str | None
    external_urls: ExternalUrls
    href: str
    id: str
    images: list[Image]
    name: str
    owner: PlaylistOwner
    public: bool | None
    snapshot_id: str
    tracks: PlaylistTracksRef
    type: str
    uri: str


# ---------------------------------------------------------------------------
# TypedDicts — show / episode / audiobook
# ---------------------------------------------------------------------------


class SimplifiedShowObject(TypedDict, total=False):
    available_markets: list[str]
    copyrights: list[CopyrightObject]
    description: str
    html_description: str
    explicit: bool
    external_urls: ExternalUrls
    href: str
    id: str
    images: list[Image]
    is_externally_hosted: bool
    languages: list[str]
    media_type: str
    name: str
    publisher: str
    type: str
    uri: str


class EpisodeObject(TypedDict, total=False):
    audio_preview_url: str | None
    description: str
    html_description: str
    duration_ms: int
    explicit: bool
    external_urls: ExternalUrls
    href: str
    id: str
    images: list[Image]
    is_externally_hosted: bool
    is_playable: bool
    languages: list[str]
    name: str
    release_date: str
    release_date_precision: str
    restrictions: Restrictions
    type: str
    uri: str
    show: SimplifiedShowObject


class AudiobookObject(TypedDict, total=False):
    authors: list[AuthorObject]
    available_markets: list[str]
    copyrights: list[CopyrightObject]
    description: str
    html_description: str
    explicit: bool
    external_urls: ExternalUrls
    href: str
    id: str
    images: list[Image]
    languages: list[str]
    media_type: str
    name: str
    narrators: list[NarratorObject]
    publisher: str
    type: str
    uri: str
    total_chapters: int


# ---------------------------------------------------------------------------
# TypedDicts — paging wrapper & top-level API types
# ---------------------------------------------------------------------------


class Paging(TypedDict, Generic[_T]):
    href: str
    limit: int
    next: str | None
    offset: int
    previous: str | None
    total: int
    items: list[_T]


class TokenData(TypedDict):
    access_token: str
    token_type: str
    expires_in: int


class PlaylistData(TypedDict, total=False):
    id: str
    name: str
    description: str | None
    public: bool
    collaborative: bool
    snapshot_id: str
    tracks: PlaylistTracksRef
    external_urls: ExternalUrls
    href: str
    uri: str
    images: list[Image]
    owner: PlaylistOwner


class SearchResult(TypedDict, total=False):
    tracks: Paging[TrackObject]
    albums: Paging[AlbumObject]
    artists: Paging[ArtistObject]
    playlists: Paging[SimplifiedPlaylistObject]
    shows: Paging[SimplifiedShowObject]
    episodes: Paging[EpisodeObject]
    audiobooks: Paging[AudiobookObject]


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------


def _request(method: str, url: str, headers: dict, body: bytes | None = None) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise urllib.error.HTTPError(
            exc.url, exc.code, exc.reason, exc.headers, exc.fp
        ) from exc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SpotifyClient:
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_BASE = "https://api.spotify.com/v1"

    def __init__(self, client_id: str = client_id, client_secret: str = client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def authenticate(self) -> str:
        """Client Credentials Flow. Returns and caches the access token."""
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
        data: TokenData = _request(  # type: ignore[assignment]
            "POST",
            self.TOKEN_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body=body,
        )
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data["expires_in"] - 60
        return self._token

    def _get_token(self) -> str:
        """Return cached token, refreshing if expired or missing."""
        if not self._token or time.time() >= self._token_expires_at:
            self.authenticate()
        return self._token  # type: ignore[return-value]

    def _auth_headers(self, token: str | None = None) -> dict:
        return {"Authorization": f"Bearer {token or self._get_token()}"}

    def authorize_user(
        self,
        scopes: list[str] | str = "playlist-modify-public playlist-modify-private",
        redirect_port: int = 8080,
    ) -> str:
        """Authorization Code + PKCE flow. Opens a browser for user consent
        and returns a user-scoped access token.

        The redirect URI ``http://127.0.0.1:{redirect_port}/callback`` must be
        registered in your Spotify app dashboard before calling this.
        """
        import hashlib
        import http.server
        import secrets
        import webbrowser

        if isinstance(scopes, list):
            scopes = " ".join(scopes)

        redirect_uri = f"http://127.0.0.1:{redirect_port}/callback"

        # PKCE challenge
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        state = secrets.token_urlsafe(16)

        auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(
            {
                "client_id": self.client_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": scopes,
                "state": state,
                "code_challenge_method": "S256",
                "code_challenge": code_challenge,
            }
        )

        auth_code: list[str] = []
        received_state: list[str] = []

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                auth_code.append(params.get("code", [""])[0])
                received_state.append(params.get("state", [""])[0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Authorization successful! You can close this tab.</h1></body></html>"
                )

            def log_message(self, format, *args):
                pass  # suppress server logs

        server = http.server.HTTPServer(("127.0.0.1", redirect_port), _Handler)
        server.timeout = 120

        print("Opening browser for Spotify authorization…")
        webbrowser.open(auth_url)
        server.handle_request()
        server.server_close()

        if not auth_code[0]:
            raise RuntimeError("No authorization code received.")
        if received_state[0] != state:
            raise RuntimeError("State mismatch — possible CSRF.")

        body = urllib.parse.urlencode(
            {
                "grant_type": "authorization_code",
                "code": auth_code[0],
                "redirect_uri": redirect_uri,
                "client_id": self.client_id,
                "code_verifier": code_verifier,
            }
        ).encode()

        data: TokenData = _request(  # type: ignore[assignment]
            "POST",
            self.TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=body,
        )
        self._refresh_token = data.get("refresh_token")
        return data["access_token"]

    def refresh_access_token(self, refresh_token: str) -> str:
        """Exchange a refresh token for a new user access token (no browser needed).

        Use this in automated contexts (e.g. Airflow DAGs). Obtain the initial
        refresh token by calling ``authorize_user()`` once interactively and
        reading ``sc._refresh_token``, then store it in an env var or secret.

        Args:
            refresh_token: The refresh token from a previous ``authorize_user()`` call.

        Returns:
            A fresh user-scoped access token.
        """
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            }
        ).encode()
        data: TokenData = _request(  # type: ignore[assignment]
            "POST",
            self.TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=body,
        )
        if new_refresh := data.get("refresh_token"):  # type: ignore[attr-defined]
            self._refresh_token = new_refresh
        return data["access_token"]

    def get_playlist(self, playlist_id: str) -> PlaylistData:
        """GET /v1/playlists/{playlist_id} — fetch a playlist by ID."""
        return _request(  # type: ignore[return-value]
            "GET",
            f"{self.API_BASE}/playlists/{playlist_id}",
            headers=self._auth_headers(),
        )

    def create_playlist(
        self,
        name: str,
        description: str = "",
        public: bool = True,
        collaborative: bool = False,
        user_access_token: str | None = None,
    ) -> PlaylistData:
        """POST /v1/me/playlists — create a playlist for the current user.

        Requires a user OAuth token with playlist-modify-public or
        playlist-modify-private scope. Pass ``user_access_token`` obtained
        via the Authorization Code flow; otherwise the client-credentials
        token is used (will 403 in production).

        Args:
            name: Name of the new playlist.
            description: Playlist description shown in Spotify clients.
            public: Whether the playlist appears on the user's profile.
            collaborative: Allow other users to modify the playlist
                           (requires ``public=False``).
            user_access_token: User-scoped OAuth token.
        """
        body = json.dumps(
            {
                "name": name,
                "description": description,
                "public": public,
                "collaborative": collaborative,
            }
        ).encode()
        return _request(  # type: ignore[return-value]
            "POST",
            f"{self.API_BASE}/me/playlists",
            headers={
                **self._auth_headers(user_access_token),
                "Content-Type": "application/json",
            },
            body=body,
        )

    def add_items_to_playlist(
        self,
        playlist_id: str,
        uris: list[str],
        position: int | None = None,
        user_access_token: str | None = None,
    ) -> dict:
        """POST /v1/playlists/{playlist_id}/tracks — add items to a playlist.

        Requires a user OAuth token with playlist-modify-public or
        playlist-modify-private scope.

        Args:
            playlist_id: The Spotify playlist ID.
            uris: List of Spotify URIs to add (e.g. ``["spotify:track:...", ...]``).
                  Max 100 per request.
            position: Zero-based index at which to insert the items. Appends
                      to the end if omitted.
            user_access_token: User-scoped OAuth token.

        Returns:
            Dict with ``snapshot_id`` of the updated playlist.
        """
        payload: dict = {"uris": uris}
        if position is not None:
            payload["position"] = position
        return _request(
            "POST",
            f"{self.API_BASE}/playlists/{playlist_id}/tracks",
            headers={
                **self._auth_headers(user_access_token),
                "Content-Type": "application/json",
            },
            body=json.dumps(payload).encode(),
        )

    def search(
        self,
        query: str,
        type: SearchType | str = SearchType.TRACK,
        limit: int = 20,
        offset: int = 0,
    ) -> SearchResult:
        """GET /v1/search — search for tracks, albums, artists, or playlists.

        Args:
            query: Search keywords.
            type: Item type(s) to search — use ``SearchType`` enum or a raw string.
                  Accepts comma-separated values (e.g. ``"track,album"``).
            limit: Max results per type (1-50).
            offset: Index of first result.
        """
        params = urllib.parse.urlencode(
            {"q": query, "type": str(type), "limit": limit, "offset": offset}
        )
        return _request(  # type: ignore[return-value]
            "GET",
            f"{self.API_BASE}/search?{params}",
            headers=self._auth_headers(),
        )

    def search_albums(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Paging[AlbumObject]:
        """Search for albums and return only the albums paging result."""
        result = self.search(query, type=SearchType.ALBUM, limit=limit, offset=offset)
        return result["albums"]
