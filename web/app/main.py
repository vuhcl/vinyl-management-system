"""
FastAPI application: Discogs auth, ingest, and ML component APIs.
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from web.app.routers import auth, ingest, ml

app = FastAPI(
    title="Vinyl Management System",
    description="Discogs integration, data ingest, recommender, condition classifier, price estimator",
    version="0.1.0",
)

@app.middleware("http")
async def session_username(request: Request, call_next):
    """Set request.state.username from cookie so ingest and API can use it."""
    request.state.username = request.cookies.get("username")
    return await call_next(request)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(ml.router, prefix="/api", tags=["ml"])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve simple dashboard with links to Login and Ingest."""
    return get_index_html()


def get_index_html() -> str:
    path = Path(__file__).parent / "templates" / "index.html"
    if path.exists():
        return path.read_text()
    return """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>Vinyl Management System</title></head>
    <body>
      <h1>Vinyl Management System</h1>
      <p><a href="/login">Log in with Discogs</a></p>
      <p><a href="/ingest">Sync / Ingest data</a></p>
      <p><a href="/docs">API docs</a></p>
    </body>
    </html>
    """


# Optional: mount static files if you add a frontend later
# static_dir = Path(__file__).parent / "static"
# if static_dir.exists():
#     app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
