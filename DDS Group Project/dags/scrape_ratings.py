import asyncio
import hashlib
import httpx

from bs4 import BeautifulSoup

from airflow import DAG
from airflow.sdk import task, TaskGroup

from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
from datetime import datetime, timedelta, timezone

from db.mongo import get_db

DELAY = 0.1

def generate_rating_id(album_id, username):
    raw = f"{album_id}_{username}"
    return hashlib.sha3_256(raw.encode()).hexdigest()


def bulk_upsert_ratings(ratings):
    db = get_db()
    operations = []
    for r in ratings:
        operations.append(
            UpdateOne(
                {"rating_id": r["rating_id"]},
                {"$set": r},
                upsert=True
            )
        )
    if operations:
        db.user_ratings.bulk_write(operations, ordered=False)

        
def parse_rating_date(date_str):
    dt = datetime.strptime(date_str,"%d %b %Y %H:%M:%S GMT")
    return dt.replace(tzinfo=timezone.utc)

        
async def scrape_user_rating_page(html):
    soup = BeautifulSoup(html, "lxml")
    rating_entries = soup.find_all("div", class_="userRatingBlock")
    results = []
    for entry in rating_entries:
        score_tag = entry.find("div", class_="rating")
        score = score_tag.text.strip() if score_tag else "N/A"
        username = entry.find("div", class_="userName").find("a")["title"]
        date = parse_rating_date(entry.find("div", class_="date")["title"])

        results.append(
            {"username": username, "score": score, "timestamp": date}
        )
    return results


async def scrape_user_ratings(slug, p):
    base_url = f"https://www.albumoftheyear.org/album/{slug}/user-reviews/?type=ratings&sort=recent"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{base_url}&p={p}", headers=headers)
        r.raise_for_status()
        results = await scrape_user_rating_page(r.text)
        await asyncio.sleep(DELAY)
    return results

default_args = {
    "owner": "airflow",
    "retries": 5,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="scrape_ratings",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_tasks=8,
    default_args=default_args,
    description="Select albums to scrape, get their new user ratings and add them to MongoDB",
    tags=["scraping"]
) as dag:
    
    @task
    def compute_album_priority():
        db = get_db()
        albums = list(db.albums.find({}))
        now = datetime.now().replace(tzinfo=None)
        for a in albums:
            release = a.get("release_date")
            if not release:
                continue
            days = (now - release).days
            if days < 0:
                continue
            priority = 1 / (days + 1)
            db.albums.update_one(
                {"_id": a["_id"]},
                {"$set": {"priority_score": priority}}
            )
            
    @task
    def select_albums():
        db = get_db()
        albums = list(db.albums.find({}, {"_id": 0}).sort("priority_score", -1).limit(1000))
        return albums
    
    with TaskGroup(group_id="scrape_ratings_group") as scrape_group:
        @task
        def scrape_ratings(album):
            db = get_db()
            slug = album["slug"]
            album_id = album["album_id"]
            last_known_posted = album.get("latest_rating_posted_at")
            newest_timestamp = None
            collected = []
            page = 1

            while True:
                ratings = asyncio.run(scrape_user_ratings(slug, page))
                if not ratings:
                    break
                stop_scraping = False
                for r in ratings:
                    rating_id = generate_rating_id(album_id, r["username"])
                    timestamp = r["timestamp"]
                    if newest_timestamp is None:
                        newest_timestamp = timestamp
                    if last_known_posted is not None and timestamp <= last_known_posted:
                        stop_scraping = True
                        break
                    doc = {
                        "rating_id": rating_id,
                        "album_id": album_id,
                        "slug": slug,
                        "username": r["username"],
                        "score": r["score"],
                        "timestamp": r["timestamp"],
                        "scraped_at": datetime.now(tz=timezone.utc)
                    }
                    collected.append(doc)
                if stop_scraping:
                    break
                page += 1
                
            bulk_upsert_ratings(collected)
                
            db.albums.update_one(
                {"_id": album_id},
                {"$set": {"last_ratings_scrape": datetime.now(tz=timezone.utc),
                          "latest_rating_posted_at": newest_timestamp}}
            )
            return len(collected)
        
        albums = select_albums()
        scrape_ratings.expand(album=albums)
        
    priority = compute_album_priority()
    albums = select_albums()
priority >> albums >> scrape_group
    