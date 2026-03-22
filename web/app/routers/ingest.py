"""
Ingest API: trigger Discogs (and optional AOTY) data sync for the logged-in user.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from core.auth import get_user_token
from core.jobs import run_discogs_ingest, run_full_ingest

router = APIRouter()


def _get_username_and_token(request: Request) -> tuple[str, str]:
    username = getattr(request.state, "username", None)
    if not username:
        raise HTTPException(
            status_code=401,
            detail="Not logged in. Use /auth/login and submit your Discogs token.",
        )
    token = get_user_token(username)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="No token for this user. Log in again at /auth/login.",
        )
    return username, token


@router.post("/sync")
async def sync_discogs(request: Request):
    """
    Fetch collection and wantlist from Discogs for the logged-in user and write to data/raw.
    """
    username, token = _get_username_and_token(request)
    try:
        result = run_discogs_ingest(username, token, per_user_dir=False)
        return {
            "username": username,
            "message": "Sync complete",
            "collection_csv": str(result["collection_csv"]),
            "wantlist_csv": str(result["wantlist_csv"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full")
async def full_ingest(request: Request):
    """
    Run full ingest (Discogs + AOTY if configured) for the logged-in user.
    """
    username, token = _get_username_and_token(request)
    try:
        result = run_full_ingest(username=username, token=token, write_csv=True)
        raw = result["raw"]
        return {
            "username": username,
            "data_dir": str(result["data_dir"]),
            "counts": {
                "collection": len(raw["collection"]),
                "wantlist": len(raw["wantlist"]),
                "ratings": len(raw["ratings"]),
                "albums": len(raw["albums"]),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_class=HTMLResponse)
async def ingest_page(request: Request):
    """Simple page to trigger sync (requires login)."""
    username = getattr(request.state, "username", None)
    if not username:
        return """
        <!DOCTYPE html><html><head><meta charset="utf-8"><title>Ingest</title></head>
        <body><h1>Sync your Discogs data</h1><p><a href="/auth/login">Log in</a> first.</p></body></html>
        """
    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>Ingest</title></head>
    <body>
      <h1>Sync your Discogs data</h1>
      <p>Logged in as <strong>{username}</strong>. <a href="/auth/login">Change account</a></p>
      <button id="sync">Sync collection & wantlist</button>
      <pre id="out"></pre>
      <script>
        document.getElementById('sync').onclick = function() {{
          fetch('/ingest/sync', {{ method: 'POST' }})
            .then(r => r.json()).then(d => document.getElementById('out').textContent = JSON.stringify(d, null, 2))
            .catch(e => document.getElementById('out').textContent = e.message);
        }};
      </script>
    </body></html>
    """
