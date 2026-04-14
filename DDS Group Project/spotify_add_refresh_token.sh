#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Write the refresh token to a temp file to avoid mixing it with
# the "Opening browser..." stdout from authorize_user()
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

echo "Running Spotify OAuth flow (a browser window will open)..."
python3 -c "
import spotify
sc = spotify.SpotifyClient()
sc.authorize_user()
with open('$TMPFILE', 'w') as f:
    f.write(sc._refresh_token or '')
"

REFRESH_TOKEN=$(cat "$TMPFILE")

if [ -z "$REFRESH_TOKEN" ] || [ "$REFRESH_TOKEN" = "None" ]; then
    echo "Error: No refresh token obtained." >&2
    exit 1
fi

echo "Storing SPOTIFY_REFRESH_TOKEN in Airflow..."
docker compose exec airflow-apiserver airflow variables set SPOTIFY_REFRESH_TOKEN "$REFRESH_TOKEN"
echo "Done. You can now trigger the spotify_token_healthcheck DAG."
