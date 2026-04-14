# create gcp bucket for storing data msds-music-recommendation-pipeline with gcloud
gcloud storage buckets create gs://msds-music-recommendation-pipeline --location=us-central1

# upload data to the bucket from ./scraping/data

# ./scraping/data/subsets/cleaned/album_info_cleaned.tsv
gcloud storage cp ./scraping/data/subsets/cleaned/album_info_cleaned.tsv gs://msds-music-recommendation-pipeline/

# ./scraping/data/subsets/cleaned/critic_ratings_cleaned.tsv
gcloud storage cp ./scraping/data/subsets/cleaned/critic_ratings_cleaned.tsv gs://msds-music-recommendation-pipeline/

# ./scraping/data/subsets/cleaned/user_ratings_cleaned.tsv
gcloud storage cp ./scraping/data/subsets/cleaned/user_ratings_cleaned.tsv gs://msds-music-recommendation-pipeline/

# ./scraping/data/subsets/raw/album_info_subset.csv
gcloud storage cp ./scraping/data/subsets/raw/album_info_subset.csv gs://msds-music-recommendation-pipeline/

# ./scraping/data/subsets/raw/critic_ratings_subset.csv
gcloud storage cp ./scraping/data/subsets/raw/critic_ratings_subset.csv gs://msds-music-recommendation-pipeline/

# ./scraping/data/subsets/raw/user_ratings_subset.csv
gcloud storage cp ./scraping/data/subsets/raw/user_ratings_subset.csv gs://msds-music-recommendation-pipeline/
