from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from recommender.src.data.discogs_aoty_id_matching import (
    DiscogsHttpHelper,
    DiscogsMatchConfig,
    build_discogs_master_to_aoty_album_id_map,
    map_discogs_release_ids_to_aoty_album_ids,
)


@dataclass
class FakeResponse:
    payload: dict[str, Any] | list[Any]
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict[str, Any] | list[Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, release_master_map: dict[str, str]):
        self.headers: dict[str, str] = {}
        self._release_master_map = release_master_map

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        _ = timeout
        if url.endswith("/database/search"):
            assert params is not None
            # Use release_title to pick search candidates.
            title = str(params.get("release_title") or "")
            if title == "Not Here Not Gone":
                # Two candidates:
                # - one year mismatch (2020)
                # - one year match (2026) -> should win
                return FakeResponse(
                    {
                        "results": [
                            {
                                "id": 111,
                                "title": "Not Here Not Gone",
                                "year": 2020,
                            },
                            {
                                "id": 222,
                                "title": "Not Here Not Gone",
                                "year": 2026,
                            },
                        ]
                    }
                )
            return FakeResponse({"results": []})

        if "/releases/" in url:
            release_id = url.rsplit("/", 1)[-1]
            master_id = self._release_master_map.get(release_id)
            return FakeResponse(
                {"master_id": int(master_id) if master_id else None}
            )

        return FakeResponse({})


def test_build_discogs_master_to_aoty_album_id_map_year_tiebreak() -> None:
    aoty_albums = pd.DataFrame(
        [
            {
                "album_id": "aoty_1",
                "artist": "Blackwater Holylight",
                "album_title": "Not Here Not Gone",
                "year": 2026,
            },
        ]
    )

    session = FakeSession(release_master_map={})
    cfg = DiscogsMatchConfig(
        min_fuzzy_similarity=0.8,
        max_results=5,
        min_request_interval_s=0,
    )
    http = DiscogsHttpHelper(session, "fake-token", cfg, cache_dir=None)

    master_map = build_discogs_master_to_aoty_album_id_map(
        aoty_albums,
        http=http,
        cfg=cfg,
    )

    assert master_map == {"222": "aoty_1"}


def test_map_discogs_release_ids_to_aoty_album_ids_filters_unmapped() -> None:
    # Create a Discogs-style df with release IDs in album_id.
    collection = pd.DataFrame(
        [
            {"user_id": "u1", "album_id": "5001"},
            {"user_id": "u1", "album_id": "5002"},
        ]
    )

    discogs_master_to_aoty = {
        "222": "aoty_1",
        # 999 intentionally missing -> should be filtered out.
    }
    session = FakeSession(release_master_map={"5001": "222", "5002": "999"})
    cfg = DiscogsMatchConfig(min_request_interval_s=0)
    http = DiscogsHttpHelper(session, "fake-token", cfg, cache_dir=None)

    out = map_discogs_release_ids_to_aoty_album_ids(
        collection,
        discogs_master_to_aoty=discogs_master_to_aoty,
        http=http,
        cfg=cfg,
    )

    # Only 5001 should remain, mapped to aoty_1
    assert len(out) == 1
    assert out.iloc[0]["album_id"] == "aoty_1"


def test_discogs_http_helper_disk_cache_roundtrip(tmp_path) -> None:
    """Persist release + search JSON caches across helper instances."""
    cfg = DiscogsMatchConfig(min_request_interval_s=0)
    s = FakeSession(release_master_map={})
    h1 = DiscogsHttpHelper(s, "t", cfg, tmp_path)
    h1._release_master["500"] = "123"
    h1._search_payloads["k1"] = {"results": [{"id": 1}]}
    h1._dirty = True
    h1.save_disk()

    h2 = DiscogsHttpHelper(FakeSession(release_master_map={}), "t", cfg, tmp_path)
    assert h2._release_master["500"] == "123"
    assert h2._search_payloads["k1"]["results"][0]["id"] == 1
