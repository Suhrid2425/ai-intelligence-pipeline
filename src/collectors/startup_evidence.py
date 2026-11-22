
import asyncio
import json
from pathlib import Path

import aiohttp

from src.collectors.startup import is_ai_candidate

YC_API_URL = "https://yc-oss.github.io/api/companies/all.json"

CANDIDATE_FILE = Path("data/raw/yc_startup_candidates.jsonl")
OUTPUT_FILE = Path("data/raw/yc_startup_evidence.jsonl")


async def main():
    with CANDIDATE_FILE.open(encoding="utf-8") as file:
        candidates = [json.loads(line) for line in file if line.strip()]

    candidate_urls = {
        record["source"]["url"]
        for record in candidates
    }

    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(YC_API_URL) as response:
            response.raise_for_status()
            companies = await response.json()

    evidence_records = []
    seen_urls = set()

    for company in companies:
        url = company.get("url")

        if url not in candidate_urls or url in seen_urls:
            continue

        if not is_ai_candidate(company):
            continue

        evidence_records.append({
            "entityName": company.get("name"),
            "sourceUrl": url,
            "oneLiner": company.get("one_liner"),
            "description": company.get("long_description"),
            "tags": company.get("tags") or [],
            "industries": company.get("industries") or [],
            "subindustry": company.get("subindustry"),
            "teamSize": company.get("team_size"),
            "profileApi": company.get("api"),
        })

        seen_urls.add(url)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in evidence_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Original candidates:", len(candidates))
    print("Evidence records:", len(evidence_records))
    print("Evidence file:", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
