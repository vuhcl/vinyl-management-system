import random
from datetime import datetime, timedelta

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

BUCKET = "aoty_data"
RATINGS_PATH = "user_ratings.tsv"
USER_EMB_PATH = "embeddings/user_embeddings.npy"
ITEM_EMB_PATH = "embeddings/item_embeddings.npy"
USER_MAP_PATH = "embeddings/user_id_mapping.json"
ITEM_MAP_PATH = "embeddings/item_id_mapping.json"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def compute_and_store(**context):
    import gcsfs
    import json
    import sys
    from datetime import datetime as dt

    import numpy as np
    import pandas as pd
    from airflow.providers.google.cloud.hooks.gcs import GCSHook
    from airflow.sdk.bases.hook import BaseHook
    from pymongo import MongoClient

    sys.path.insert(0, "/opt/airflow")
    from recommender import get_recommendations

    n_users = context["params"]["n_users"]

    gcs_hook = GCSHook(gcp_conn_id="google_cloud_default")
    credentials, project_id = gcs_hook.get_credentials_and_project_id()
    fs = gcsfs.GCSFileSystem(project=project_id, token=credentials)

    with fs.open(f"{BUCKET}/{USER_EMB_PATH}", "rb") as f:
        user_embeddings = np.load(f)
    with fs.open(f"{BUCKET}/{ITEM_EMB_PATH}", "rb") as f:
        item_embeddings = np.load(f)
    with fs.open(f"{BUCKET}/{USER_MAP_PATH}") as f:
        user_id_mapping = json.load(f)
    with fs.open(f"{BUCKET}/{ITEM_MAP_PATH}") as f:
        item_id_mapping = json.load(f)
    with fs.open(f"{BUCKET}/{RATINGS_PATH}") as f:
        user_ratings = pd.read_csv(f, sep="\t")

    user_ids = user_ratings["username"].map(user_id_mapping).values
    item_ids = user_ratings["slug"].map(item_id_mapping).values

    all_usernames = list(user_id_mapping.keys())
    sample = random.sample(all_usernames, min(n_users, len(all_usernames)))
    print(f"Computing recommendations for {len(sample)} users")

    conn = BaseHook.get_connection("mongo_default")
    client = MongoClient(f"mongodb://{conn.host}:{conn.port}")
    db = client[conn.schema or "aoty"]
    collection = db["user_recommendations"]

    for username in sample:
        recs = get_recommendations(
            username,
            user_embeddings,
            item_embeddings,
            user_id_mapping,
            item_id_mapping,
            user_ids,
            item_ids,
        )
        collection.update_one(
            {"username": username},
            {"$set": {
                "recommendations": [{"slug": s, "score": score} for s, score in recs],
                "updated_at": dt.utcnow(),
            }},
            upsert=True,
        )

    print(f"Upserted recommendations for {len(sample)} users into user_recommendations")
    client.close()


with DAG(
    "batch_recommend",
    default_args=default_args,
    description="Compute top-20 recommendations for N random users and store in MongoDB",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "spotify"],
    params={"n_users": 100},  # type: ignore[arg-type]
) as dag:
    PythonOperator(
        task_id="compute_and_store",
        python_callable=compute_and_store,
    )
