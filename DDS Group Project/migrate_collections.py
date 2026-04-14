import hashlib
from datetime import datetime, timezone
from pymongo import UpdateOne
from db.mongo import get_db

db = get_db()

BATCH_SIZE = 1000


def split_slug(slug_value):
    parts = slug_value.split("-", 1)
    album_id = int(parts[0])
    new_slug = parts[1] if len(parts) > 1 else ""
    return album_id, new_slug


def migrate_slugs(collection_name):
    collection = db[collection_name]
    operations = []
    count = 0
    for doc in collection.find():
        old_slug = doc["slug"]
        album_id, new_slug = split_slug(old_slug)

        operations.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"album_id": album_id}},
            )
        )

        if len(operations) >= BATCH_SIZE:
            collection.bulk_write(operations, ordered=False)
            count += len(operations)
            print(f"  Processed {count} documents...")
            operations = []

    if operations:
        collection.bulk_write(operations, ordered=False)
        count += len(operations)

    print(f"  Updated {count} documents in {collection_name}")


def migrate_genres():
    collection = db.albums
    operations = []
    count = 0
    for doc in collection.find():
        genres_raw = doc.get("genres", "")
        if isinstance(genres_raw, str):
            genres = [g.strip() for g in genres_raw.split("|") if g.strip()]
        else:
            continue 

        operations.append(UpdateOne({"_id": doc["_id"]},{"$set": {"genres": genres}},))
        if len(operations) >= BATCH_SIZE:
            collection.bulk_write(operations, ordered=False)
            count += len(operations)
            print(f"  Processed {count} documents...")
            operations = []
    if operations:
        collection.bulk_write(operations, ordered=False)
        count += len(operations)
    print(f"  Updated {count} genres in albums")


def generate_rating_id(album_id, username):
    raw = f"{album_id}_{username}"
    return hashlib.sha3_256(raw.encode()).hexdigest()


def parse_rating_date(date_str):
    dt = datetime.strptime(date_str,"%d %b %Y %H:%M:%S GMT")
    return dt.replace(tzinfo=timezone.utc)


def migrate_rating_ids():
    collection = db.user_ratings
    operations = []
    count = 0
    for doc in collection.find():
        date = doc.get("date")
        if date:
            parsed = parse_rating_date(date)
        else:
            parsed = doc.get("timestamp")
        album_id = doc.get("album_id")
        if album_id is None:
            album_id, _ = split_slug(doc["slug"])
        rating_id = generate_rating_id(album_id, doc["username"])

        operations.append(UpdateOne({"_id": doc["_id"]}, 
                                    {"$set": {"rating_id": rating_id, "timestamp": parsed},
                                     "$unset": {"date": ""}}))
        if len(operations) >= BATCH_SIZE:
            collection.bulk_write(operations, ordered=False)
            count += len(operations)
            print(f"  Processed {count} documents...")
            operations = []
    if operations:
        collection.bulk_write(operations, ordered=False)
        count += len(operations)
    print(f"  Updated {count} rating_ids in user_ratings")


def parse_release_date(date_string):
    for fmt in ("%B %d %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    return None


def migrate_release_dates():
    collection = db.albums
    operations = []
    count = 0
    skipped = 0
    for doc in collection.find():
        operations.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"latest_rating_posted_at": None, "last_ratings_scrape": None}}
                )
            )
        release_date_raw = doc.get("release_date", "")
        release_year = doc.get("release_year", "")
        combined = f"{release_date_raw} {release_year}".strip()
        parsed = parse_release_date(combined)
        if parsed is None:
            skipped += 1
            continue

        operations.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {
                    "$set": {"release_date_string": combined, "release_date": parsed},
                    "$unset": {"release_year": ""},
                },
            )
        )            
        if len(operations) >= BATCH_SIZE:
            collection.bulk_write(operations, ordered=False)
            count += len(operations)
            print(f"  Processed {count} documents...")
            operations = []
    if operations:
        collection.bulk_write(operations, ordered=False)
        count += len(operations)
    if skipped:
        print(f"  Skipped {skipped} documents (could not parse date)")
    print(f"  Updated {count} release dates in albums")
    
    
def create_index():
    db.user_ratings.create_index("rating_id", unique=True)
    db.user_ratings.create_index(
        [("album_id", 1), ("timestamp", -1)]
    )
    db.user_ratings.create_index("user_id")
    db.user_ratings.create_index("timestamp")

    db.albums.create_index("slug", unique=True)
    db.albums.create_index("last_ratings_scrape")


if __name__ == "__main__":
    for name in ["albums", "critic_ratings", "user_ratings"]:
        migrate_slugs(name)
    print("Slug migration complete!")

    migrate_genres()
    print("Genres migration complete!")

    migrate_rating_ids()
    print("Rating ID migration complete!")

    migrate_release_dates()
    print("Release date migration complete!")
    
    create_index()
    print("Indexing complete!")
