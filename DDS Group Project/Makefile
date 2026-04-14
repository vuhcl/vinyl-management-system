init:
  ## download credentials json and save it in setup/gcp/credentials.json
	gcloud iam service-accounts keys create setup/gcp/credentials.json --iam-account=airflow-sa@music-recommendation-pipeline.iam.gserviceaccount.com

  # Download data files from GCP
  echo "Downloading data files from GCP..."
  GCS_BUCKET="gs://msds-music-recommendation-pipeline"

  gcloud storage cp "$GCS_BUCKET/album_info_cleaned.tsv" /tmp/album_info_cleaned.tsv
  gcloud storage cp "$GCS_BUCKET/critic_ratings_cleaned.tsv" /tmp/critic_ratings_cleaned.tsv
  gcloud storage cp "$GCS_BUCKET/user_ratings_cleaned.tsv" /tmp/user_ratings_cleaned.tsv

  #Import data into MongoDB
  mongoimport --host mongodb --db music --collection albums --type tsv --headerline --file /tmp/album_info_cleaned.tsv || true
  mongoimport --host mongodb --db music --collection critic_ratings --type tsv --headerline --file /tmp/critic_ratings_cleaned.tsv || true
  mongoimport --host mongodb --db music --collection user_ratings --type tsv --headerline --file /tmp/user_ratings_cleaned.tsv || true
