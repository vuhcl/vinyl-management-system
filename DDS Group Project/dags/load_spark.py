import os
import socket
from datetime import datetime, timedelta, timezone

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

# gs://aoty_data/user_ratings.tsv
GCS_BUCKET = "aoty_data"
GCS_FILE = "user_ratings.tsv"


def top_rated_albums_last_day(**context):
    from airflow.sdk.bases.hook import BaseHook
    from pymongo import MongoClient
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import FloatType, StringType, StructField, StructType

    run_gcp = os.environ.get("RUN_GCP", "false").lower() == "true"
    print(f"run_gcp={run_gcp}")

    # to run locally
    if not run_gcp:
        # Load Spark master URL from Airflow connection (conn_type=spark, host=spark-master, port=7077)
        spark_conn = BaseHook.get_connection("spark_default")
        spark_master = f"spark://{spark_conn.host}:{spark_conn.port or 7077}"

        gcs_jar = "/opt/gcp/gcs-connector-hadoop3-latest.jar"
        if not os.path.exists(gcs_jar):
            raise FileNotFoundError(f"GCS connector JAR not found at {gcs_jar}")

        spark = (
            SparkSession.builder.appName("top_rated_albums")
            .master(spark_master)
            .config("spark.driver.host", socket.gethostbyname(socket.gethostname()))
            .config("spark.jars", gcs_jar)
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
            .config(
                "spark.hadoop.google.cloud.auth.service.account.json.keyfile",
                "/opt/gcp/credentials.json",
            )
            .config(
                "spark.hadoop.fs.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            )
            .config(
                "spark.hadoop.fs.AbstractFileSystem.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
            )
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("top_rated_albums").getOrCreate()

    schema = StructType(
        [
            StructField("slug", StringType(), nullable=False),
            StructField("username", StringType(), nullable=False),
            StructField("score", FloatType(), nullable=False),
            StructField("date", StringType(), nullable=False),
        ]
    )

    spark_df = spark.read.csv(
        f"gs://{GCS_BUCKET}/{GCS_FILE}",
        sep="\t",
        header=True,
        schema=schema,
    )
    print(f"Loaded {spark_df.count()} rows from GCS")

    # Parse date column: "18 Aug 2025 13:24:26 GMT"
    spark_df = spark_df.withColumn(
        "parsed_date",
        F.to_timestamp(F.col("date"), "dd MMM yyyy HH:mm:ss z"),
    )

    # Filter to last 365 days relative to execution date
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    recent_df = spark_df.filter(F.col("parsed_date") >= F.lit(cutoff))

    # Use Spark SQL to find highest rated albums
    recent_df.createOrReplaceTempView("ratings")
    top_albums = spark.sql("""
        SELECT
            slug,
            COUNT(*)          AS rating_count,
            ROUND(AVG(score), 2) AS avg_score
        FROM ratings
        GROUP BY slug
        ORDER BY avg_score DESC, rating_count DESC
        LIMIT 20
    """)

    top_albums.show(truncate=False)
    results = top_albums.collect()
    spark.stop()

    # Build result document
    document = {
        "computed_at": datetime.now(timezone.utc),
        "window": "last_1_day",
        "cutoff": cutoff,
        "top_albums": [
            {
                "slug": row["slug"],
                "avg_score": row["avg_score"],
                "rating_count": row["rating_count"],
            }
            for row in results
        ],
    }

    # Insert into MongoDB
    mongo_conn = BaseHook.get_connection("mongo_default")
    client = MongoClient(host=mongo_conn.host, port=mongo_conn.port or 27017)
    db = client[mongo_conn.schema or "msds"]
    collection = db["top_rated_albums"]
    result = collection.insert_one(document)
    print(f"Inserted document id: {result.inserted_id}")
    client.close()

    return str(result.inserted_id)


with DAG(
    "load_spark",
    default_args=default_args,
    description="Find top-rated albums in last day via Spark SQL and insert into MongoDB",
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["spark"],
) as dag:
    top_albums_task = PythonOperator(
        task_id="top_rated_albums_last_day",
        python_callable=top_rated_albums_last_day,
    )
