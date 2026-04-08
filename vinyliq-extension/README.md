# VinylIQ Chrome extension

Thin client for the **VinylIQ price microservice** (`price_estimator/src/api/main.py`).

## Setup

1. Start the API (from monorepo root):

   ```bash
   PYTHONPATH=. uvicorn price_estimator.src.api.main:app --host 127.0.0.1 --port 8801
   ```

2. Chrome → Extensions → Load unpacked → select this `vinyliq-extension` folder.

3. Open any `https://www.discogs.com/release/<id>` page, click the extension icon, set **API base URL** (default `http://127.0.0.1:8801`), optional **API key** if the server has `VINYLIQ_API_KEY` set.

4. Choose media/sleeve grades and **Get estimate**. Results appear in a fixed overlay.

## Permissions

- `https://www.discogs.com/*` — content script + release ID
- `http://127.0.0.1/*`, `http://localhost/*`, `https://*/*` — configurable API host (broad `https://*/*` so you can point to a deployed API; tighten for production builds)
