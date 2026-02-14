"""
Discogs authentication: token-based login and session.
"""
import os
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.auth import set_user_token, get_user_token, get_token_for_request

router = APIRouter()


class TokenSubmit(BaseModel):
    token: str


def get_current_username(request: Request) -> str | None:
    """Resolve current user from request state (set after login)."""
    return getattr(request.state, "username", None)


@router.post("/token")
async def submit_token(
    body: TokenSubmit,
    request: Request,
):
    """
    Verify Discogs token and store for this session. Returns username and sets cookie.
    """
    token = (body.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token required")
    try:
        from discogs_api import DiscogsClient
        client = DiscogsClient(user_token=token)
        username = client.get_username()
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token or Discogs API error")
        set_user_token(username, token)
        request.state.username = username
        response = JSONResponse(content={"username": username, "message": "Logged in"})
        response.set_cookie(key="username", value=username, httponly=True, samesite="lax", max_age=86400 * 7)
        return response
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Discogs auth failed: {e}")


@router.get("/me")
async def me(request: Request):
    """Return current username if logged in (from cookie, set by middleware)."""
    username = getattr(request.state, "username", None) or request.cookies.get("username")
    return {"username": username, "logged_in": username is not None}


@router.get("/login", response_class=HTMLResponse)
async def login_page():
    """Simple login page: user pastes Discogs token."""
    return """
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>Log in with Discogs</title></head>
    <body>
      <h1>Log in with Discogs</h1>
      <p>Get a personal access token from
        <a href="https://www.discogs.com/settings/developers" target="_blank">Discogs → Settings → Developers</a>
        and paste it below.
      </p>
      <form id="f" action="/auth/token" method="post">
        <label>Token <input type="password" name="token" placeholder="Your Discogs token" size="40" /></label>
        <button type="submit">Log in</button>
      </form>
      <p>We only use your token to fetch your collection and wantlist; we do not store it permanently.</p>
      <script>
        document.getElementById('f').addEventListener('submit', function(e) {
          e.preventDefault();
          var form = e.target;
          var fd = new FormData(form);
          fetch(form.action, { method: 'POST', body: JSON.stringify({ token: fd.get('token') }), headers: { 'Content-Type': 'application/json' } })
            .then(r => r.json()).then(function(d) { alert('Logged in as ' + d.username); window.location.href = '/'; })
            .catch(function() { alert('Login failed'); });
        });
      </script>
    </body></html>
    """
