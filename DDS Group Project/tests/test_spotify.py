import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from spotify import PlaylistData, SearchResult, SearchType, SpotifyClient, TokenData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: dict, status: int = 200):
    """Return a mock that behaves like urllib.request.urlopen's context manager."""
    raw = json.dumps(data).encode()
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = raw
    mock.status = status
    return mock


TOKEN_RESPONSE: TokenData = {
    "access_token": "test-token",
    "token_type": "Bearer",
    "expires_in": 3600,
}
PLAYLIST_RESPONSE: PlaylistData = {
    "id": "pl123",
    "name": "My Playlist",
    "tracks": {"items": []},
}
SEARCH_RESPONSE: SearchResult = {"tracks": {"items": [{"id": "t1", "name": "Song"}]}}
CREATE_RESPONSE: PlaylistData = {"id": "pl_new", "name": "New Playlist"}


# ---------------------------------------------------------------------------
# SearchType enum
# ---------------------------------------------------------------------------


def test_search_type_values():
    assert SearchType.TRACK == "track"
    assert SearchType.ALBUM == "album"
    assert SearchType.ARTIST == "artist"
    assert SearchType.PLAYLIST == "playlist"
    assert SearchType.SHOW == "show"
    assert SearchType.EPISODE == "episode"
    assert SearchType.AUDIOBOOK == "audiobook"


def test_search_type_is_str_subclass():
    assert isinstance(SearchType.TRACK, str)


def test_search_type_str_conversion():
    assert str(SearchType.ALBUM) == "album"


# ---------------------------------------------------------------------------
# authenticate()
# ---------------------------------------------------------------------------


def test_authenticate_returns_token():
    client = SpotifyClient("id", "secret")
    with patch("urllib.request.urlopen", return_value=_mock_response(TOKEN_RESPONSE)):
        token = client.authenticate()
    assert token == "test-token"
    assert client._token == "test-token"


def test_authenticate_raises_on_http_error():
    client = SpotifyClient("id", "bad-secret")
    http_err = urllib.error.HTTPError(
        url=None, code=401, msg="Unauthorized", hdrs=None, fp=BytesIO(b"")
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            client.authenticate()
    assert exc_info.value.code == 401


# ---------------------------------------------------------------------------
# _get_token() auto-refresh
# ---------------------------------------------------------------------------


def test_get_token_auto_refreshes_when_expired():
    client = SpotifyClient("id", "secret")
    client._token = "old-token"
    client._token_expires_at = 0.0  # already expired

    with patch("urllib.request.urlopen", return_value=_mock_response(TOKEN_RESPONSE)):
        token = client._get_token()

    assert token == "test-token"


def test_get_token_reuses_valid_token():
    client = SpotifyClient("id", "secret")
    client._token = "cached-token"
    client._token_expires_at = 9_999_999_999.0  # far future

    with patch("urllib.request.urlopen") as mock_open:
        token = client._get_token()

    mock_open.assert_not_called()
    assert token == "cached-token"


# ---------------------------------------------------------------------------
# get_playlist()
# ---------------------------------------------------------------------------


def test_get_playlist_returns_playlist_data():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(PLAYLIST_RESPONSE)
    ):
        result = client.get_playlist("pl123")

    assert result["id"] == "pl123"
    assert result["name"] == "My Playlist"


def test_get_playlist_raises_on_404():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    http_err = urllib.error.HTTPError(
        url=None, code=404, msg="Not Found", hdrs=None, fp=BytesIO(b"")
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            client.get_playlist("nonexistent")
    assert exc_info.value.code == 404


# ---------------------------------------------------------------------------
# create_playlist()
# ---------------------------------------------------------------------------


def test_create_playlist_posts_to_me_endpoint():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(CREATE_RESPONSE)
    ) as mock_open:
        result = client.create_playlist("New Playlist", description="desc", public=True)

    assert result["name"] == "New Playlist"
    req = mock_open.call_args[0][0]
    assert req.full_url.endswith("/me/playlists")
    body = json.loads(req.data)
    assert body == {
        "name": "New Playlist",
        "description": "desc",
        "public": True,
        "collaborative": False,
    }


def test_create_playlist_collaborative():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(CREATE_RESPONSE)
    ) as mock_open:
        client.create_playlist("Collab List", public=False, collaborative=True)

    body = json.loads(mock_open.call_args[0][0].data)
    assert body["collaborative"] is True
    assert body["public"] is False


def test_create_playlist_uses_user_access_token():
    client = SpotifyClient("id", "secret")
    client._token = "client-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(CREATE_RESPONSE)
    ) as mock_open:
        client.create_playlist("My List", user_access_token="user-oauth-token")

    req = mock_open.call_args[0][0]
    assert req.get_header("Authorization") == "Bearer user-oauth-token"


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_default_type_is_track():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(SEARCH_RESPONSE)
    ) as mock_open:
        result = client.search("Radiohead")

    assert "tracks" in result
    req = mock_open.call_args[0][0]
    assert "q=Radiohead" in req.full_url
    assert "type=track" in req.full_url


def test_search_with_enum_type():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(SEARCH_RESPONSE)
    ) as mock_open:
        client.search("OK Computer", type=SearchType.ALBUM)

    req = mock_open.call_args[0][0]
    assert "type=album" in req.full_url


def test_search_with_raw_string_type():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(SEARCH_RESPONSE)
    ) as mock_open:
        client.search("Miles Davis", type="artist")

    req = mock_open.call_args[0][0]
    assert "type=artist" in req.full_url


def test_search_limit_and_offset():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(SEARCH_RESPONSE)
    ) as mock_open:
        client.search("jazz", limit=5, offset=10)

    req = mock_open.call_args[0][0]
    assert "limit=5" in req.full_url
    assert "offset=10" in req.full_url


# ---------------------------------------------------------------------------
# search_albums()
# ---------------------------------------------------------------------------


def test_search_albums_returns_paging():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    albums_paging = {
        "href": "https://api.spotify.com/v1/search",
        "limit": 20,
        "next": None,
        "offset": 0,
        "previous": None,
        "total": 1,
        "items": [{"id": "alb1", "name": "OK Computer", "album_type": "album"}],
    }
    response = {"albums": albums_paging}

    with patch("urllib.request.urlopen", return_value=_mock_response(response)) as mock_open:
        result = client.search_albums("OK Computer")

    assert result["total"] == 1
    assert result["items"][0]["name"] == "OK Computer"
    req = mock_open.call_args[0][0]
    assert "type=album" in req.full_url
    assert "q=OK+Computer" in req.full_url


def test_search_albums_forwards_limit_and_offset():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    albums_paging = {"href": "", "limit": 5, "next": None, "offset": 10, "previous": None, "total": 0, "items": []}
    with patch("urllib.request.urlopen", return_value=_mock_response({"albums": albums_paging})) as mock_open:
        client.search_albums("jazz", limit=5, offset=10)

    req = mock_open.call_args[0][0]
    assert "limit=5" in req.full_url
    assert "offset=10" in req.full_url


# ---------------------------------------------------------------------------
# add_items_to_playlist()
# ---------------------------------------------------------------------------

ADD_ITEMS_RESPONSE = {"snapshot_id": "snap123"}


def test_add_items_posts_to_correct_endpoint():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(ADD_ITEMS_RESPONSE)
    ) as mock_open:
        result = client.add_items_to_playlist("pl123", ["spotify:track:abc"])

    assert result["snapshot_id"] == "snap123"
    req = mock_open.call_args[0][0]
    assert req.full_url.endswith("/playlists/pl123/tracks")
    assert req.get_method() == "POST"


def test_add_items_sends_uris_in_body():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    uris = ["spotify:track:aaa", "spotify:track:bbb"]
    with patch(
        "urllib.request.urlopen", return_value=_mock_response(ADD_ITEMS_RESPONSE)
    ) as mock_open:
        client.add_items_to_playlist("pl123", uris)

    body = json.loads(mock_open.call_args[0][0].data)
    assert body["uris"] == uris
    assert "position" not in body


def test_add_items_sends_position_when_provided():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(ADD_ITEMS_RESPONSE)
    ) as mock_open:
        client.add_items_to_playlist("pl123", ["spotify:track:aaa"], position=0)

    body = json.loads(mock_open.call_args[0][0].data)
    assert body["position"] == 0


def test_add_items_uses_user_access_token():
    client = SpotifyClient("id", "secret")
    client._token = "client-token"
    client._token_expires_at = 9_999_999_999.0

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(ADD_ITEMS_RESPONSE)
    ) as mock_open:
        client.add_items_to_playlist(
            "pl123", ["spotify:track:aaa"], user_access_token="user-oauth-token"
        )

    req = mock_open.call_args[0][0]
    assert req.get_header("Authorization") == "Bearer user-oauth-token"


def test_add_items_raises_on_401():
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    http_err = urllib.error.HTTPError(
        url=None, code=401, msg="Unauthorized", hdrs=None, fp=BytesIO(b"")
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            client.add_items_to_playlist("pl123", ["spotify:track:aaa"])
    assert exc_info.value.code == 401


# ---------------------------------------------------------------------------
# refresh_access_token()
# ---------------------------------------------------------------------------

REFRESH_RESPONSE = {
    "access_token": "new-access-token",
    "token_type": "Bearer",
    "expires_in": 3600,
}


def test_refresh_access_token_returns_new_access_token():
    client = SpotifyClient("id", "secret")

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(REFRESH_RESPONSE)
    ):
        token = client.refresh_access_token("my-refresh-token")

    assert token == "new-access-token"


def test_refresh_access_token_posts_correct_grant_type():
    client = SpotifyClient("id", "secret")

    with patch(
        "urllib.request.urlopen", return_value=_mock_response(REFRESH_RESPONSE)
    ) as mock_open:
        client.refresh_access_token("my-refresh-token")

    import urllib.parse
    body = urllib.parse.parse_qs(mock_open.call_args[0][0].data.decode())
    assert body["grant_type"] == ["refresh_token"]
    assert body["refresh_token"] == ["my-refresh-token"]
    assert body["client_id"] == ["id"]


def test_refresh_access_token_stores_new_refresh_token_when_rotated():
    client = SpotifyClient("id", "secret")
    rotated = {**REFRESH_RESPONSE, "refresh_token": "rotated-refresh-token"}

    with patch("urllib.request.urlopen", return_value=_mock_response(rotated)):
        client.refresh_access_token("old-refresh-token")

    assert client._refresh_token == "rotated-refresh-token"


def test_refresh_access_token_does_not_overwrite_refresh_token_when_absent():
    client = SpotifyClient("id", "secret")
    client._refresh_token = "original"

    with patch("urllib.request.urlopen", return_value=_mock_response(REFRESH_RESPONSE)):
        client.refresh_access_token("original")

    assert client._refresh_token == "original"


def test_refresh_access_token_raises_on_400():
    client = SpotifyClient("id", "secret")

    http_err = urllib.error.HTTPError(
        url=None, code=400, msg="Bad Request", hdrs=None, fp=BytesIO(b"")
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            client.refresh_access_token("invalid-token")
    assert exc_info.value.code == 400


# ---------------------------------------------------------------------------
# search() — parametrized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("search_type", list(SearchType))
def test_search_accepts_all_enum_values(search_type: SearchType):
    client = SpotifyClient("id", "secret")
    client._token = "test-token"
    client._token_expires_at = 9_999_999_999.0

    with patch("urllib.request.urlopen", return_value=_mock_response({})) as mock_open:
        client.search("test", type=search_type)

    req = mock_open.call_args[0][0]
    assert f"type={search_type.value}" in req.full_url
