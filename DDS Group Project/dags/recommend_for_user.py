from datetime import datetime, timedelta

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def build_playlist(**context):
    import sys

    from airflow.sdk import Variable
    from airflow.sdk.bases.hook import BaseHook
    from pymongo import MongoClient

    sys.path.insert(0, "/opt/airflow")
    from recommender import slug_to_search_query
    from spotify import SpotifyClient

    username = context["params"]["username"]

    # Read pre-computed recommendations from MongoDB
    conn = BaseHook.get_connection("mongo_default")
    client = MongoClient(f"mongodb://{conn.host}:{conn.port}")
    db = client[conn.schema or "aoty"]
    doc = db["user_recommendations"].find_one({"username": username})
    client.close()

    if not doc:
        raise ValueError(
            f"No recommendations found for '{username}'. Run batch_recommend first."
        )

    slugs = [r["slug"] for r in doc["recommendations"]]

    # Get Spotify access token via stored refresh token
    refresh_token = Variable.get("SPOTIFY_REFRESH_TOKEN")
    sc = SpotifyClient()
    access_token = sc.refresh_access_token(refresh_token)
    if sc._refresh_token and sc._refresh_token != refresh_token:
        Variable.set("SPOTIFY_REFRESH_TOKEN", sc._refresh_token)

    # Create playlist
    playlist = sc.create_playlist(
        name=f"AOTY Recommendations for {username}",
        description="Auto-generated album recommendations",
        public=False,
        user_access_token=access_token,
    )
    playlist_id = playlist["id"]

    # Search for a track per album slug and collect URIs
    uris = []
    for slug in slugs:
        query = slug_to_search_query(slug)
        results = sc.search(query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if items:
            uris.append(items[0]["uri"])

    if uris:
        sc.add_items_to_playlist(playlist_id, uris, user_access_token=access_token)

    print(f"Created playlist '{playlist['name']}' with {len(uris)} tracks")
    print(f"Playlist URL: {playlist.get('external_urls', {}).get('spotify', '')}")


with DAG(
    "recommend_for_user",
    default_args=default_args,
    description="Build a Spotify playlist from pre-computed MongoDB recommendations for a user",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "spotify"],
    params={"username": ""},  # type: ignore[arg-type]
) as dag:
    PythonOperator(
        task_id="build_playlist",
        python_callable=build_playlist,
    )
