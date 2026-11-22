import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "news.jsonl"

FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AIIntelligencePipeline/1.0)",
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
}

TIMEOUT = aiohttp.ClientTimeout(total=35)


def clean_text(value):
    if not value:
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(value, "html.parser").get_text(" ")).strip()


def parse_date(value):
    if not value:
        return None

    try:
        result = parsedate_to_datetime(value)
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)
    except Exception:
        pass

    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)
    except Exception:
        return None


def tag_text(element, names):
    for name in names:
        found = element.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return ""


def parse_feed(xml_text, source):
    root = ET.fromstring(xml_text)
    entries = []

    # RSS format
    for item in root.findall(".//item"):
        link = tag_text(item, ["link"])
        title = tag_text(item, ["title"])
        published = (
            tag_text(item, ["pubDate", "published", "date"])
        )
        description = tag_text(
            item, ["description", "content:encoded"]
        )

        entries.append({
            "title": clean_text(title),
            "url": link,
            "published": published,
            "summary": clean_text(description),
        })

    # Atom format
    if not entries:
        for item in root.findall(".//{*}entry"):
            title = tag_text(item, ["{*}title"])

            link = ""
            for element in item.findall("{*}link"):
                href = element.attrib.get("href")
                rel = element.attrib.get("rel", "alternate")
                if href and rel == "alternate":
                    link = href
                    break

            if not link:
                for element in item.findall("{*}link"):
                    if element.attrib.get("href"):
                        link = element.attrib["href"]
                        break

            published = (
                tag_text(item, ["{*}published", "{*}updated"])
            )
            summary = (
                tag_text(item, ["{*}summary", "{*}content"])
            )

            entries.append({
                "title": clean_text(title),
                "url": link,
                "published": published,
                "summary": clean_text(summary),
            })

    return entries


async def fetch_feed(session, source, feed_url):
    try:
        async with session.get(feed_url) as response:
            response.raise_for_status()
            body = await response.text()
            return source, parse_feed(body, source)

    except Exception as exc:
        logger.warning("Feed failed: %s: %s", source, exc)
        return source, []


async def extract_full_text(session, url):
    if not url:
        return ""

    try:
        async with session.get(url) as response:
            response.raise_for_status()
            html = await response.text()

        soup = BeautifulSoup(html, "lxml")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "aside", "form", "noscript"
        ]):
            tag.decompose()

        article = soup.find("article")

        if article:
            text = article.get_text(" ", strip=True)
        else:
            main = soup.find("main")
            text = (
                main.get_text(" ", strip=True)
                if main else soup.get_text(" ", strip=True)
            )

        return re.sub(r"\s+", " ", text).strip()

    except Exception as exc:
        logger.warning("Article extraction failed: %s: %s", url, exc)
        return ""


async def collect_news():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

    connector = aiohttp.TCPConnector(limit=10)

    async with aiohttp.ClientSession(
        headers=HEADERS,
        timeout=TIMEOUT,
        connector=connector,
    ) as session:

        feed_tasks = [
            fetch_feed(session, source, url)
            for source, url in FEEDS.items()
        ]

        feed_results = await asyncio.gather(*feed_tasks)

        candidates = []
        seen_urls = set()

        for source, entries in feed_results:
            print(f"{source}: {len(entries)} feed entries")

            for entry in entries:
                published = parse_date(entry["published"])
                url = urljoin(
                    FEEDS[source], entry.get("url", "")
                )

                if not entry["title"] or not url:
                    continue

                if not published or published < cutoff or published > now:
                    continue

                normalized_url = url.split("#")[0].rstrip("/")

                if normalized_url in seen_urls:
                    continue

                seen_urls.add(normalized_url)

                candidates.append({
                    "source": source,
                    "title": entry["title"],
                    "url": normalized_url,
                    "publishedAt": published.isoformat(),
                    "summary": entry["summary"],
                })

        print(f"\nArticles within last 24 hours: {len(candidates)}")
        print("Extracting article full text...")

        semaphore = asyncio.Semaphore(8)

        async def process(item):
            async with semaphore:
                text = await extract_full_text(
                    session, item["url"]
                )

            item["fullText"] = text
            item["fullTextExtracted"] = bool(text)
            item["collectedAt"] = now.isoformat()
            return item

        results = await asyncio.gather(
            *(process(item) for item in candidates)
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    extracted = sum(x["fullTextExtracted"] for x in results)

    print("\n========== NEWS COLLECTION REPORT ==========")
    print("Configured sources:", len(FEEDS))
    print("Sources responding:", sum(bool(items) for _, items in feed_results))
    print("Recent articles:", len(results))
    print("Full text extracted:", extracted)
    print("Full text missing:", len(results) - extracted)
    print("Output:", OUTPUT_FILE)
    print("============================================")


if __name__ == "__main__":
    asyncio.run(collect_news())