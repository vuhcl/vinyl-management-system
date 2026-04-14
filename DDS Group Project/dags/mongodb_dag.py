from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk.bases.hook import BaseHook
from datetime import datetime
from pymongo import MongoClient


def get_mongo_client():
    """
    Create a MongoDB client using Airflow connection.
    """
    conn = BaseHook.get_connection("mongo_default")

    uri = f"mongodb://{conn.host}:{conn.port}"
    client = MongoClient(uri)

    return client, conn.schema


def insert_document(**context):
    client, db_name = get_mongo_client()
    db = client[db_name]
    collection = db["users"]

    doc = {
        "user_id": 1234,
        "name": "Mahesh Chaudhari",
        "role": "data_engineer",
        "created_at": datetime.utcnow(),
    }

    result = collection.insert_one(doc)
    print(f"Inserted document id: {result.inserted_id}")

    client.close()


def find_documents(**context):
    client, db_name = get_mongo_client()
    db = client[db_name]
    collection = db["users"]

    results = collection.find({"role": "data_engineer"})
    for doc in results:
        print(doc)

    client.close()


def update_document(**context):
    client, db_name = get_mongo_client()
    db = client[db_name]
    collection = db["users"]

    result = collection.update_one(
        {"user_id": 1234},
        {"$set": {"role": "senior_data_engineer"}}
    )

    print(f"Matched: {result.matched_count}, Modified: {result.modified_count}")
    client.close()


def delete_document(**context):
    client, db_name = get_mongo_client()
    db = client[db_name]
    collection = db["users"]

    result = collection.delete_one({"user_id": 1234})
    print(f"Deleted documents: {result.deleted_count}")

    client.close()


with DAG(
    dag_id="mongodb_basic_operations",
    start_date=datetime(2024, 1, 1),
    schedule=None,              # manual trigger
    catchup=False,
    tags=["mongodb", "airflow", "example"],
) as dag:

    insert_task = PythonOperator(
        task_id="insert_document",
        python_callable=insert_document,
    )

    find_task = PythonOperator(
        task_id="find_documents",
        python_callable=find_documents,
    )

    update_task = PythonOperator(
        task_id="update_document",
        python_callable=update_document,
    )

    delete_task = PythonOperator(
        task_id="delete_document",
        python_callable=delete_document,
    )

    insert_task >> find_task >> update_task >> delete_task
