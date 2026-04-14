import asyncio
import httpx

from bs4 import BeautifulSoup

from airflow import DAG
from airflow.sdk import task, TaskGroup

from datetime import datetime, timedelta

from db.mongo import get_db

DELAY = 0.3

async def scrape_album_slug(html):
    results = []
    soup = BeautifulSoup(html, "lxml")
    for block in soup.select("div.albumBlock.small"):
        if len(block.select("div.ratingRow")) >= 1:
            slug = block.select_one('a[href^="/album"]')["href"][7:-4]
            results.append(slug)
    return results

async def parse_album(html):
    soup = BeautifulSoup(html, "lxml")
    artist = soup.find("div", class_="artist").text
    album = soup.find("h1", class_="albumTitle").text
    detail_row = soup.select_one("div.albumTopBox.info div.detailRow")
    texts = list(detail_row.stripped_strings)
    month = texts[0]
    date = texts[1].replace(",", "")
    year = texts[2]
    release_date_str = f"{month} {date} {year}"
    try:
        release_date = datetime.strptime(release_date_str, "%B %d %Y")
    except ValueError:
        release_date = datetime.strptime(release_date_str, "%B %Y")
    genre_metas = soup.select('div.detailRow meta[itemprop="genre"]')
    genres = [m["content"] for m in genre_metas]
    return {
        "artist": artist,
        "album": album,
        "release_date": release_date,
        "release_date_string": release_date_str,
        "genres": genres,
        "latest_rating_posted_at": None,
        "last_ratings_scrape": None
    }

async def scrape_album_info(slug):
    url = f"https://www.albumoftheyear.org/album/{slug}.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, follow_redirects=True)
    album = await parse_album(r.text)
    album["slug"] = slug
    parts = slug.split("-", 1)
    album["album_id"] = int(parts[0])
    return album

async def scrape_new_releases():
    base = "https://www.albumoftheyear.org/releases/this-week/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"
    }
    results = []
    page_num = 0
    stop_flag = True
    while stop_flag:
        page_num += 1
        url = f"{base}{page_num}"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, follow_redirects=True)
        data = await scrape_album_slug(r.text)
        if len(data) > 0:
            results.extend(data)
        if (len(data) < 40) or (len(results) > 100):
            stop_flag = False
    return results

default_args = {
    "owner": "airflow",
    "retries": 5,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="scrape_new_albums",
    start_date=datetime(2024, 1, 1),
    schedule="0 12 * * FRI",
    catchup=False,
    max_active_tasks=10,
    default_args=default_args,
    tags=["scraping"],
    description="Get new albums released this week and add them to MongoDB"
) as dag:
    
    @task
    def run_scrape_new_releases():
        results = asyncio.run(scrape_new_releases())
        return results
    
    with TaskGroup(group_id="scrape_albums_group") as scrape_group:
        @task
        def scrape_album(slug):
            db = get_db()
            a = asyncio.run(scrape_album_info(slug))
            db.albums.update_one(
                {"album_id": a["album_id"]},
                {"$set": a},
                upsert=True
            )
        
        slugs = run_scrape_new_releases()
        scrape_album.expand(slug=slugs)
        
    slugs = run_scrape_new_releases()
slugs >> scrape_group
    
    