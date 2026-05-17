# VinylIQ Chrome extension

Thin client for two VinylIQ FastAPI services:

- **Price API** (`price_estimator/src/api/main.py`) — `POST /estimate`,
  rendered as a floating overlay on `https://www.discogs.com/release/*`.
- **Grader API** (`grader/serving/main.py`) — `POST /predict`, wired on seller drafts
  `https://www.discogs.com/sell/post/*`: a **floating dock** on the listing page stays open while you work (Chrome toolbar popups close on outside click).

Version `0.2.0` split the legacy single `apiBase` into `priceApiBase` /
`graderApiBase`; **`0.3.x`** adds the seller-page dock next to drafts and
mirrors slim actions in the toolbar popup.

## Setup

1. Start both APIs (from monorepo root):

   ```bash
   # Price API on 8801
   PYTHONPATH=. uvicorn price_estimator.src.api.main:app \
     --host 127.0.0.1 --port 8801

   # Grader API on 8090 (needs MLFLOW_MODEL_URI and tracking URI in env)
   PYTHONPATH=. uvicorn grader.serving.main:app \
     --host 127.0.0.1 --port 8090
   ```

2. Chrome → Extensions → Load unpacked → select this `vinyliq-extension`
   folder. **Details** shows the extension ID (depends on unpack path).

   Reference ID for the maintained VinylIQ unpack in this monorepo:
   **`bhhebpplkmapokijgbmeejhamdlkbcao`** (use **`chrome://extensions`** if Chrome lists a different one).

3. Click the extension icon to open the popup, set:

   - **Price API base URL** (default `http://127.0.0.1:8801`)
   - **Grader API base URL** (default `http://127.0.0.1:8090`)
   - Optional **API key** (sent as `X-API-Key` to both services)

4. Two flows are supported:

   - **Release page** (`/release/<id>`): pick media/sleeve grades and
     click **Get estimate**. Results appear in a fixed overlay.
   - **Seller listing** (`/sell/post/<release-id>`): once the VinylIQ dock appears bottom-right,
     type or paste the seller note in **Condition comments** (`#comments` or textarea fallbacks),
     click **Grade condition**, then **Get estimate** when you want a price overlay / suggested USD.
     Toolbar popup repeats the flow if you prefer it—popup focus does not persist while clicking Discogs controls.

## GKE demo deployment

When pointed at the demo Gateway, set both bases to the same host with
service-specific path prefixes:

- Price: `https://<ip>.nip.io/price`
- Grader: `https://<ip>.nip.io/grader`

The Gateway's HTTPRoute strips the prefix before forwarding to the
upstream, so the inner FastAPI routes (`/estimate`, `/predict`) match
unchanged.

## Permissions (manifest 0.2.0)

`host_permissions` was tightened in 0.2.0:

- `https://www.discogs.com/*` — content scripts + release ID parsing
- `http://127.0.0.1/*`, `http://localhost/*` — local API hosts
- `https://*.nip.io/*` — GKE demo Gateway (replace with the real
  hostname before publishing)

The previous `https://*/*` catch-all is removed; tighten further for
non-demo builds.
