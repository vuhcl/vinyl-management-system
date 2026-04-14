from pymongo import MongoClient
import os

def get_mongo_client():
    uri = "mongodb://localhost:27017"
    # uri = os.environ.get("MONGO_URI")
    return MongoClient(uri)

def get_db():
    client = get_mongo_client()
    return client["music"]