
import argparse
import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from src.config import settings
from src.schemas.entities import (
    ResearchPaperEntity,
    ResearchPaperContent,
)


logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"

ATOM_NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom"
}

REQUEST_DELAY_SECONDS = 3


class ArxivCollector:
    """Collect AI research papers from the official arXiv API."""

    def __init__(self):
        self.batch_size = settings.arxiv_batch_size
        self.timeout = settings.request_timeout
        self.max_retries = settings.max_retries

    async def fetch_page(
        self,
        session: aiohttp.ClientSession,
        start: int,
        max_results: int,
    ) -> str:
        """Fetch one page of research paper metadata with retries."""

        params = {
            "search_query": "cat:cs.AI",
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(
                    ARXIV_API_URL,
                    params=params,
                ) as response:

                    # Retry rate limits and temporary server errors.
                    if response.status == 429 or response.status >= 500:
                        if attempt == self.max_retries:
                            response.raise_for_status()

                        retry_after = response.headers.get("Retry-After")

                        try:
                            delay = float(retry_after) if retry_after else 2 ** attempt
                        except ValueError:
                            delay = 2 ** attempt

                        delay = min(max(delay, 0), 30)

                        logger.warning(
                            "arXiv returned HTTP %s. Retrying in %.1f seconds.",
                            response.status,
                            delay,
                        )

                        await asyncio.sleep(delay)
                        continue

                    # Do not retry permanent client errors such as 400 or 403.
                    response.raise_for_status()

                    return await response.text()

            except aiohttp.ClientResponseError as exc:
                if exc.status != 429 and exc.status < 500:
                    raise

                if attempt == self.max_retries:
                    raise

                delay = min(2 ** attempt, 30)

                logger.warning(
                    "HTTP error %s. Retrying in %s seconds.",
                    exc.status,
                    delay,
                )

                await asyncio.sleep(delay)

            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == self.max_retries:
                    raise

                delay = min(2 ** attempt, 30)

                logger.warning(
                    "Network request failed. Retry %s/%s in %s seconds.",
                    attempt + 1,
                    self.max_retries,
                    delay,
                )

                await asyncio.sleep(delay)

        raise RuntimeError("Unable to fetch arXiv page after retries.")

    @staticmethod
    def parse_datetime(value: str) -> datetime:
        """Convert an arXiv ISO timestamp into a timezone-aware datetime."""

        parsed = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    def parse_feed(
        self,
        xml_content: str,
    ) -> list[ResearchPaperEntity]:
        """Parse Atom XML into validated research paper entities."""

        root = ET.fromstring(xml_content)

        entries = root.findall("atom:entry", ATOM_NAMESPACE)

        papers = []

        for entry in entries:
            try:
                paper_id = entry.findtext(
                    "atom:id",
                    namespaces=ATOM_NAMESPACE,
                )

                title = entry.findtext(
                    "atom:title",
                    namespaces=ATOM_NAMESPACE,
                )

                published = entry.findtext(
                    "atom:published",
                    namespaces=ATOM_NAMESPACE,
                )

                authors = [
                    author.findtext(
                        "atom:name",
                        namespaces=ATOM_NAMESPACE,
                    )
                    for author in entry.findall(
                        "atom:author",
                        ATOM_NAMESPACE,
                    )
                ]

                if not all([paper_id, title, published]):
                    logger.warning("Skipping incomplete arXiv entry.")
                    continue

                # Extract the arXiv identifier and remove only
                # a trailing version suffix such as v1 or v2.
                identifier = paper_id.rstrip("/").split("/")[-1]
                identifier = re.sub(r"v\d+$", "", identifier)

                canonical_url = (
                    f"https://arxiv.org/abs/{identifier}"
                )

                paper = ResearchPaperEntity(
                    content=ResearchPaperContent(
                        title=" ".join(title.split()),
                        authors=[
                            author.strip()
                            for author in authors
                            if author and author.strip()
                        ],
                        paper_url=canonical_url,
                        github_url=None,
                        github_stars=None,
                        published_date=self.parse_datetime(published),
                    )
                )

                papers.append(paper)

            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Invalid arXiv record skipped: %s",
                    exc,
                )

        return papers

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path,
    ) -> dict[str, ResearchPaperEntity]:
        """Load previously collected and validated papers from JSONL."""

        existing_papers = {}

        if not checkpoint_path.exists():
            logger.info("No existing checkpoint found.")
            return existing_papers

        logger.info("Loading checkpoint from %s", checkpoint_path)

        with checkpoint_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    paper = ResearchPaperEntity.model_validate(data)

                    key = str(paper.content.paper_url)

                    existing_papers[key] = paper

                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Skipping invalid checkpoint line %s: %s",
                        line_number,
                        exc,
                    )

        logger.info(
            "Loaded %s existing papers from checkpoint.",
            len(existing_papers),
        )

        return existing_papers

    @staticmethod
    def save_checkpoint(
        papers: dict[str, ResearchPaperEntity],
        checkpoint_path: Path,
    ) -> None:
        """Atomically save collected papers to a JSONL checkpoint."""

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for paper in papers.values():
                file.write(
                    json.dumps(
                        paper.model_dump(mode="json"),
                        ensure_ascii=False,
                    ) + "\n"
                )

        # Replace the previous checkpoint only after writing completes.
        temporary_path.replace(checkpoint_path)

        logger.info(
            "Checkpoint saved: %s papers.",
            len(papers),
        )

    async def collect(
        self,
        limit: int,
        checkpoint_path: Path,
    ) -> list[ResearchPaperEntity]:
        """Collect unique papers and save progress after every page."""

        unique_papers = self.load_checkpoint(checkpoint_path)

        # If the checkpoint already meets the requested limit,
        # no API request is necessary.
        if len(unique_papers) >= limit:
            logger.info(
                "Checkpoint already contains enough papers: %s/%s",
                len(unique_papers),
                limit,
            )

            return list(unique_papers.values())[:limit]

        timeout = aiohttp.ClientTimeout(
            total=self.timeout
        )

        headers = {
            "User-Agent": (
                "AIIntelligencePipeline/1.0 "
                "(research project)"
            )
        }

        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
        ) as session:

            start = 0

            while len(unique_papers) < limit:

                remaining = limit - len(unique_papers)

                batch_size = min(
                    self.batch_size,
                    remaining,
                )

                logger.info(
                    "Fetching arXiv records: start=%s, batch=%s",
                    start,
                    batch_size,
                )

                xml_content = await self.fetch_page(
                    session=session,
                    start=start,
                    max_results=batch_size,
                )

                page_papers = self.parse_feed(xml_content)

                if not page_papers:
                    logger.info("No more papers returned by arXiv.")
                    break

                new_records = 0

                for paper in page_papers:
                    key = str(paper.content.paper_url)

                    if key not in unique_papers:
                        unique_papers[key] = paper
                        new_records += 1

                # Save progress immediately after processing each page.
                self.save_checkpoint(
                    papers=unique_papers,
                    checkpoint_path=checkpoint_path,
                )

                logger.info(
                    "Page processed: %s records, %s new records.",
                    len(page_papers),
                    new_records,
                )

                logger.info(
                    "Unique papers collected: %s/%s",
                    len(unique_papers),
                    limit,
                )

                if len(page_papers) < batch_size:
                    logger.info("Reached the end of available results.")
                    break

                start += batch_size

                if len(unique_papers) < limit:
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)

        results = list(unique_papers.values())[:limit]

        logger.info(
            "Collection finished with %s unique papers.",
            len(results),
        )

        return results


async def main(limit: int):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_path = Path("data/raw/arxiv_papers.jsonl")

    collector = ArxivCollector()

    papers = await collector.collect(
        limit=limit,
        checkpoint_path=output_path,
    )

    print("\n========== COLLECTION SUMMARY ==========")
    print(f"Requested papers : {limit}")
    print(f"Collected papers : {len(papers)}")
    print(f"Output file      : {output_path}")
    print("=========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect AI research papers from arXiv."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=settings.target_papers,
        help="Maximum number of unique papers to collect.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    asyncio.run(main(args.limit))
