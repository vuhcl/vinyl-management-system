from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# https://storage.googleapis.com/aoty_data/user_ratings.tsv
# https://storage.googleapis.com/aoty_data/album_info.tsv
# https://storage.googleapis.com/aoty_data/critic_ratings.tsv
files = [
    ("gs://aoty_data/user_ratings.tsv", "user_ratings"),
    ("gs://aoty_data/album_info.tsv", "album_info"),
    ("gs://aoty_data/critic_ratings.tsv", "critic_ratings"),
]


def make_load_task(gcs_uri: str, collection_name: str):
    def load_file(**context):
        import gcsfs
        import pandas as pd
        from airflow.providers.google.cloud.hooks.gcs import GCSHook
        from airflow.sdk.bases.hook import BaseHook
        from pymongo import MongoClient

        # Strip gs:// prefix for gcsfs
        path = gcs_uri.removeprefix("gs://")

        gcs_hook = GCSHook(gcp_conn_id="google_cloud_default")
        credentials, project_id = gcs_hook.get_credentials_and_project_id()
        fs = gcsfs.GCSFileSystem(project=project_id, token=credentials)
        with fs.open(path) as f:
            df = pd.read_csv(f, sep="\t")

        print(f"Loaded {len(df)} rows from {gcs_uri}")

        conn = BaseHook.get_connection("mongo_default")
        client = MongoClient(host=conn.host, port=conn.port or 27017)
        db = client[conn.schema or "msds"]
        collection = db[collection_name]

        records = df.to_dict(orient="records")
        result = collection.insert_many(records)
        print(f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'")
        client.close()

    load_file.__name__ = f"load_{collection_name}"
    return load_file


with DAG(
    "load_music_data",
    default_args=default_args,
    description="Load GCS files into separate MongoDB collections",
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    tasks = [
        PythonOperator(
            task_id=f"load_{collection_name}",
            python_callable=make_load_task(gcs_uri, collection_name),
        )
        for gcs_uri, collection_name in files
    ]
