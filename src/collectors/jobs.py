
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import aiohttp
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "jobs.jsonl"

SOURCES = {
    "Remote OK": "https://remoteok.com/api",
    "Remotive": "https://remotive.com/api/remote-jobs",
    "Arbeitnow": "https://www.arbeitnow.com/api/job-board-api",
    "Jobicy": "https://jobicy.com/api/jobs?count=50",
    "Himalayas": "https://himalayas.app/jobs/api?limit=20",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AIIntelligencePipeline/1.0)",
    "Accept": "application/json,text/html,*/*",
}

TIMEOUT = aiohttp.ClientTimeout(total=45, connect=15)

logger = logging.getLogger(__name__)

AI_PATTERN = re.compile(
    r"\b(ai engineer|artificial intelligence|machine learning|"
    r"deep learning|data scientist|data science|llm|"
    r"large language model|generative ai|nlp|"
    r"natural language processing|computer vision|"
    r"robotics|prompt engineer|ai researcher|"
    r"research scientist|ml engineer|"
    r"machine learning engineer|python engineer|"
    r"rag engineer|ai developer|ai architect|"
    r"mlops|mle|genai|intelligent automation)\b",
    re.IGNORECASE,
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(str(item) for item in value)

    if isinstance(value, dict):
        value = " ".join(str(v) for v in value.values())

    return BeautifulSoup(
        str(value), "html.parser"
    ).get_text(" ", strip=True).strip()


def parse_date(value: Any) -> datetime | None:
    if value is None or value == "":
        return None

    try:
        if isinstance(value, (int, float)):
            timestamp = float(value)

            if timestamp > 10_000_000_000:
                timestamp /= 1000

            return datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            )

        text = str(value).strip()

        # Handle ISO dates and timestamps.
        try:
            result = datetime.fromisoformat(
                text.replace("Z", "+00:00")
            )

            if result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)

            return result.astimezone(timezone.utc)

        except ValueError:
            pass

        # Handle RFC 2822 dates.
        try:
            result = parsedate_to_datetime(text)

            if result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)

            return result.astimezone(timezone.utc)

        except (TypeError, ValueError, OverflowError):
            return None

    except (ValueError, TypeError, OverflowError, OSError):
        return None


def first_value(item: dict, *keys: str) -> Any:
    for key in keys:
        value = item.get(key)

        if value is not None and value != "":
            return value

    return None


def extract_jobs(source: str, payload: Any) -> list[dict]:
    if isinstance(payload, list):
        records = payload

    elif isinstance(payload, dict):
        records = (
            payload.get("jobs")
            or payload.get("data")
            or payload.get("results")
            or []
        )

    else:
        return []

    if not isinstance(records, list):
        return []

    results = []

    for item in records:
        if not isinstance(item, dict):
            continue

        # Remote OK includes a metadata object in its response.
        if source == "Remote OK" and not item.get("position"):
            continue

        title = first_value(
            item,
            "title",
            "position",
            "jobTitle",
            "name",
        )

        company = first_value(
            item,
            "company",
            "company_name",
            "companyName",
            "organization",
            "employer",
        )

        url = first_value(
            item,
            "url",
            "job_url",
            "jobUrl",
            "applicationLink",
            "link",
        )

        description = first_value(
            item,
            "description",
            "jobDescription",
            "description_text",
            "excerpt",
            "jobExcerpt",
            "content",
        )

        published = first_value(
            item,
            "publication_date",
            "pubDate",
            "created_at",
            "date",
            "published_at",
            "published",
            "postedDate",
            "updated_at",
        )

        # Source-specific date fields.
        if source == "Remote OK":
            published = first_value(
                item, "date", "epoch", "publication_date"
            )

        elif source == "Remotive":
            published = first_value(
                item, "publication_date", "date"
            )

        elif source == "Arbeitnow":
            published = first_value(
                item, "created_at", "updated_at"
            )

        elif source == "Jobicy":
            published = first_value(
                item, "pubDate", "published_at", "date"
            )

        elif source == "Himalayas":
            published = first_value(
                item, "pubDate", "published_at", "created_at"
            )

        if not title or not company or not url:
            continue

        if not str(url).startswith(("http://", "https://")):
            continue

        location = first_value(
            item,
            "location",
            "jobGeo",
            "candidate_required_location",
            "locationRestrictions",
            "candidate_location",
        )

        job_type = first_value(
            item,
            "job_type",
            "jobType",
            "employmentType",
            "job_types",
            "type",
        )

        remote_value = first_value(
            item,
            "remote",
            "is_remote",
            "remote_allowed",
            "isRemote",
        )

        remote_text = str(location or "").lower()

        remote = (
            remote_value is True
            or str(remote_value).lower() in ("true", "1", "yes")
            or "remote" in remote_text
        )

        results.append({
            "source": source,
            "title": clean_text(title),
            "company": clean_text(company),
            "url": str(url).strip(),
            "published": parse_date(published),
            "description": clean_text(description),
            "location": clean_text(location),
            "employmentType": clean_text(job_type),
            "remote": remote,
            "rawId": str(
                first_value(item, "id", "guid", "slug") or ""
            ),
        })

    return results


async def fetch_json(
    session: aiohttp.ClientSession,
    source: str,
    url: str,
) -> Any:
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json(content_type=None)

    except Exception as exc:
        logger.warning("%s API failed: %s", source, exc)
        return None


async def extract_full_text(
    session: aiohttp.ClientSession,
    job: dict,
) -> str:
    description = job["description"]

    # Prefer the description supplied by the API.
    if len(description) >= 300:
        return description

    try:
        async with session.get(job["url"]) as response:
            response.raise_for_status()
            html = await response.text()

        soup = BeautifulSoup(html, "lxml")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
        ]):
            tag.decompose()

        container = (
            soup.find("article")
            or soup.find("main")
            or soup.body
            or soup
        )

        extracted = clean_text(container.get_text(" ", strip=True))

        # Avoid replacing useful API text with a shorter page extract.
        return extracted if len(extracted) > len(description) else description

    except Exception as exc:
        logger.warning(
            "Description extraction failed for %s: %s",
            job["url"],
            exc,
        )
        return description


async def collect_jobs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

    connector = aiohttp.TCPConnector(limit=10)

    async with aiohttp.ClientSession(
        headers=HEADERS,
        timeout=TIMEOUT,
        connector=connector,
    ) as session:

        tasks = [
            fetch_json(session, source, url)
            for source, url in SOURCES.items()
        ]

        payloads = await asyncio.gather(*tasks)

        all_jobs = []
        source_counts = {}

        for (source, _), payload in zip(SOURCES.items(), payloads):
            jobs = extract_jobs(source, payload) if payload is not None else []

            source_counts[source] = len(jobs)
            all_jobs.extend(jobs)

        candidates = []
        seen = set()

        for job in all_jobs:
            published = job["published"]

            # Strict 24-hour filter. Do not invent missing dates.
            if published is None or not (cutoff <= published <= now):
                continue

            searchable = job["title"] + " " + job["description"]

            if not AI_PATTERN.search(searchable):
                continue

            normalized_url = job["url"].split("#")[0].rstrip("/")

            key = normalized_url.lower()

            if not key:
                key = (
                    job["company"].lower(),
                    job["title"].lower(),
                )

            if key in seen:
                continue

            seen.add(key)
            job["url"] = normalized_url
            candidates.append(job)

        print("\nJobs inside the 24-hour window:", len(candidates))
        print("Extracting full descriptions...")

        semaphore = asyncio.Semaphore(8)

        async def process(job: dict) -> dict:
            async with semaphore:
                full_text = await extract_full_text(session, job)

            return {
                "recordType": "JOB",
                "schemaVersion": "1.0",
                "source": {
                    "name": job["source"],
                    "url": job["url"],
                },
                "content": {
                    "company": job["company"],
                    "title": job["title"],
                    "date": job["published"].isoformat(),
                    "is_remote": job["remote"],
                    "location": job["location"],
                    "employmentType": job["employmentType"],
                    "role_family": "AI",
                    "description": full_text,
                    "fullTextExtracted": bool(full_text),
                    "sourceJobId": job["rawId"] or None,
                },
                "collectedAt": now.isoformat(),
            }

        results = await asyncio.gather(
            *(process(job) for job in candidates)
        )

    # Write atomically to avoid leaving a partially written output file.
    temporary_output = OUTPUT_FILE.with_suffix(".jsonl.tmp")

    with temporary_output.open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    temporary_output.replace(OUTPUT_FILE)

    print("\n========== JOB COLLECTION REPORT ==========")

    for source, count in source_counts.items():
        print(f"{source}: {count} source records")

    print("Configured job boards:", len(SOURCES))
    print("Recent AI jobs:", len(results))
    print(
        "Full descriptions extracted:",
        sum(
            bool(item["content"]["fullTextExtracted"])
            for item in results
        ),
    )
    print("Output:", OUTPUT_FILE)
    print("============================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(collect_jobs())
