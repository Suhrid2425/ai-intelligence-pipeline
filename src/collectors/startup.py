
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from src.schemas.entities import StartupEntity

logger = logging.getLogger(__name__)

YC_API_URL = "https://yc-oss.github.io/api/companies/all.json"

AI_KEYWORDS = re.compile(
    r"\b("
    r"artificial intelligence|machine learning|generative ai|"
    r"large language models?|llms?|deep learning|computer vision|"
    r"natural language processing|nlp|ai agents?|ai-powered|ai-driven|"
    r"neural networks?|predictive analytics|speech recognition|"
    r"autonomous agents?"
    r")\b",
    re.IGNORECASE,
)

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "yc_startup_candidates.jsonl"


def is_ai_candidate(company: dict) -> bool:
    """Identify possible AI-related companies using keyword matching."""

    searchable_text = " ".join([
        str(company.get("name") or ""),
        str(company.get("one_liner") or ""),
        str(company.get("long_description") or ""),
        " ".join(company.get("tags") or []),
        " ".join(company.get("industries") or []),
        str(company.get("subindustry") or ""),
    ])

    return bool(AI_KEYWORDS.search(searchable_text))


def normalize_company(company: dict) -> StartupEntity:
    """Convert a YC company into the required startup schema."""

    return StartupEntity(
        schemaVersion="1.0",
        recordType="STARTUP",
        source={
            "name": "Y Combinator",
            "url": company["url"],
        },
        content={
            "entityName": company["name"].strip(),
            "data": {
                "employeeCount": company.get("team_size"),
            },
        },
        collectedAt=datetime.now(timezone.utc),
    )


async def fetch_yc_companies() -> list[dict]:
    """Download YC company directory with retry handling."""

    timeout = aiohttp.ClientTimeout(total=60)
    headers = {"User-Agent": "AI-Intelligence-Pipeline/1.0"}

    for attempt in range(1, 4):
        try:
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
            ) as session:
                async with session.get(YC_API_URL) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if not isinstance(data, list):
                        raise ValueError("Unexpected YC API response format")

                    return data

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning(
                "YC request attempt %s failed: %s",
                attempt,
                exc,
            )

            if attempt == 3:
                raise

            await asyncio.sleep(2 ** attempt)


def save_jsonl(records: list[StartupEntity]) -> None:
    """Save validated records as JSON Lines."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                record.model_dump_json() + "\n"
            )


async def collect_startups() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("Starting YC startup collection")

    companies = await fetch_yc_companies()
    logger.info("Downloaded %s YC companies", len(companies))

    candidates = []
    seen_names = set()

    for company in companies:
        if not is_ai_candidate(company):
            continue

        name = (company.get("name") or "").strip()
        url = company.get("url")

        if not name or not url:
            continue

        if not url.startswith("https://www.ycombinator.com/companies/"):
            continue

        normalized_name = name.casefold()

        if normalized_name in seen_names:
            continue

        try:
            record = normalize_company(company)
            candidates.append(record)
            seen_names.add(normalized_name)

        except Exception as exc:
            logger.warning(
                "Skipping invalid company %s: %s",
                name,
                exc,
            )

    save_jsonl(candidates)

    logger.info("Total YC companies: %s", len(companies))
    logger.info("AI keyword candidates: %s", len(candidates))
    logger.info("Saved records to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(collect_startups())
