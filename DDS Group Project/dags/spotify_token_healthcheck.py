from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_spotify_token(**context):
    import sys

    from airflow.sdk import Variable

    sys.path.insert(0, "/opt/airflow")
    from spotify import SpotifyClient, _request

    refresh_token = Variable.get("SPOTIFY_REFRESH_TOKEN")

    sc = SpotifyClient()
    access_token = sc.refresh_access_token(refresh_token)

    if sc._refresh_token and sc._refresh_token != refresh_token:
        Variable.set("SPOTIFY_REFRESH_TOKEN", sc._refresh_token)
        print("Rotated refresh token saved to Airflow Variable")

    profile = _request(
        "GET",
        "https://api.spotify.com/v1/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    print(
        f"Token valid. Spotify user: {profile.get('display_name')} ({profile.get('id')})"
    )


with DAG(
    "spotify_token_healthcheck",
    default_args=default_args,
    description="Validates the Spotify user OAuth token every 5 minutes",
    schedule=timedelta(minutes=5),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["spotify"],
) as dag:
    PythonOperator(
        task_id="check_token",
        python_callable=check_spotify_token,
    )
