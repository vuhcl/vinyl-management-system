#!/bin/bash
set -e

# Wait for Airflow DB to be ready and run migrations
airflow db migrate

# Create admin user
airflow users create \
  --username airflow \
  --password airflow \
  --firstname Air \
  --lastname Flow \
  --role Admin \
  --email airflow@airflow.com || true

# Add connection for the example postgres database
airflow connections add 'postgres-db-example' \
  --conn-uri 'postgresql://postgres:postgres@postgres-db-example:5432/example' || true

# Add connection for the local MongoDB database
airflow connections add 'mongo_default' \
  --conn-uri 'mongodb://mongodb:27017/msds' || true

# Add connection for GCP (uses Application Default Credentials)
airflow connections add 'google_cloud_default' \
  --conn-type 'google_cloud_platform' || true

# Add connection for Spark cluster
airflow connections add 'spark_default' \
  --conn-type 'spark' \
  --conn-host 'spark-master' \
  --conn-port '7077' || true

# Verify required packages are installed
echo "Verifying package installation..."
python3 -c "import gcsfs; print('gcsfs OK')"
python3 -c "import airflow.providers.google; print('apache-airflow-providers-google OK')"



echo "Setup complete!"
