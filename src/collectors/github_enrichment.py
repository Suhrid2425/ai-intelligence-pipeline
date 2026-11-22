
import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from pydantic import HttpUrl, ValidationError

from src.config import settings
from src.schemas.entities import ResearchPaperEntity


logger = logging.getLogger(__name__)

ARXIV_DELAY_SECONDS = 3
GITHUB_API_URL = "https://api.github.com/repos"

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "AI-Intelligence-Pipeline",
}

RESERVED_GITHUB_PATHS = {
    "features", "topics", "collections", "marketplace",
    "settings", "login", "signup", "orgs", "sponsors",
    "about", "site",
}


class GitHubEnricher:
    """Find and verify GitHub repositories linked to arXiv papers."""

    def __init__(self):
        self.timeout = settings.request_timeout
        self.max_retries = settings.max_retries
        self.github_token = getattr(settings, "github_token", "")

        if self.github_token:
            logger.info("GitHub API authentication is configured.")
        else:
            logger.warning(
                "No GitHub token found. Using unauthenticated API access."
            )

    def get_github_headers(self) -> dict:
        headers = GITHUB_HEADERS.copy()

        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        return headers

    @staticmethod
    def extract_github_repo_url(html: str) -> str | None:
        """Extract an explicit GitHub owner/repository link."""

        soup = BeautifulSoup(html, "html.parser")

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            parsed = urlparse(href)

            if parsed.hostname not in {"github.com", "www.github.com"}:
                continue

            parts = [
                part for part in parsed.path.strip("/").split("/")
                if part
            ]

            if len(parts) < 2:
                continue

            owner = parts[0]
            repository = re.sub(r"\.git$", "", parts[1])

            if owner.lower() in RESERVED_GITHUB_PATHS:
                continue

            if not owner or not repository:
                continue

            if repository in {".", ".."}:
                continue

            return f"https://github.com/{owner}/{repository}"

        return None

    @staticmethod
    def retry_delay(
        attempt: int,
        headers=None,
        default_max: int = 30,
    ) -> int:
        """Calculate a retry delay using server-provided headers."""

        headers = headers or {}

        retry_after = headers.get("Retry-After")

        if retry_after and retry_after.isdigit():
            return max(int(retry_after), 1)

        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")

        if remaining == "0" and reset and reset.isdigit():
            return max(int(reset) - int(time.time()), 1)

        return min(2 ** attempt, default_max)

    async def fetch_arxiv_page(
        self,
        session: aiohttp.ClientSession,
        paper_url: str,
    ) -> str | None:
        """Fetch an arXiv abstract page with retry handling."""

        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(paper_url) as response:

                    if response.status == 429 or response.status >= 500:
                        if attempt == self.max_retries:
                            logger.error(
                                "arXiv request failed after retries: HTTP %s",
                                response.status,
                            )
                            return None

                        delay = self.retry_delay(
                            attempt,
                            response.headers,
                        )

                        logger.warning(
                            "arXiv HTTP %s. Retrying in %s seconds.",
                            response.status,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    response.raise_for_status()
                    return await response.text()

            except aiohttp.ClientResponseError as exc:
                logger.error(
                    "arXiv HTTP error %s for %s",
                    exc.status,
                    paper_url,
                )
                return None

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == self.max_retries:
                    logger.error(
                        "arXiv network failure after retries: %s",
                        exc,
                    )
                    return None

                delay = min(2 ** attempt, 30)

                logger.warning(
                    "arXiv network error. Retrying in %s seconds: %s",
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        return None

    async def verify_github_repository(
        self,
        session: aiohttp.ClientSession,
        repository_url: str,
    ) -> tuple[tuple[str, int] | None, bool]:
        """
        Verify a public GitHub repository.

        Returns:
            (repository_data, completed)

        completed=True:
            A definitive result was obtained.

        completed=False:
            A transient failure occurred; retry in a later run.
        """

        parsed = urlparse(repository_url)

        if parsed.hostname not in {"github.com", "www.github.com"}:
            return None, True

        parts = [
            part for part in parsed.path.strip("/").split("/")
            if part
        ]

        if len(parts) != 2:
            return None, True

        owner, repository = parts
        api_url = f"{GITHUB_API_URL}/{owner}/{repository}"

        for attempt in range(self.max_retries + 1):
            try:
                async with session.get(
                    api_url,
                    headers=self.get_github_headers(),
                ) as response:

                    if response.status == 404:
                        logger.info(
                            "GitHub repository not found: %s",
                            repository_url,
                        )
                        return None, True

                    if response.status in {403, 429}:
                        remaining = response.headers.get(
                            "X-RateLimit-Remaining"
                        )
                        retry_after = response.headers.get("Retry-After")

                        is_rate_limited = (
                            response.status == 429
                            or remaining == "0"
                            or retry_after is not None
                        )

                        if not is_rate_limited:
                            logger.warning(
                                "GitHub access forbidden: %s",
                                repository_url,
                            )
                            return None, False

                        if attempt == self.max_retries:
                            logger.error(
                                "GitHub rate limit persists for %s",
                                repository_url,
                            )
                            return None, False

                        delay = self.retry_delay(
                            attempt,
                            response.headers,
                            default_max=60,
                        )

                        logger.warning(
                            "GitHub rate limit reached. "
                            "Retrying in %s seconds.",
                            delay,
                        )

                        await asyncio.sleep(delay)
                        continue

                    if response.status >= 500:
                        if attempt == self.max_retries:
                            logger.error(
                                "GitHub server error: HTTP %s",
                                response.status,
                            )
                            return None, False

                        delay = min(2 ** attempt, 30)
                        await asyncio.sleep(delay)
                        continue

                    response.raise_for_status()
                    data = await response.json()

                    canonical_url = data.get("html_url")
                    stars = data.get("stargazers_count")

                    if not isinstance(canonical_url, str):
                        return None, False

                    if not isinstance(stars, int):
                        return None, False

                    return (canonical_url, stars), True

            except aiohttp.ClientResponseError as exc:
                logger.warning(
                    "GitHub HTTP error %s for %s",
                    exc.status,
                    repository_url,
                )
                return None, False

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == self.max_retries:
                    logger.warning(
                        "GitHub network failure after retries: %s",
                        exc,
                    )
                    return None, False

                await asyncio.sleep(min(2 ** attempt, 30))

        return None, False

    async def enrich_one(
        self,
        session: aiohttp.ClientSession,
        paper: ResearchPaperEntity,
    ) -> tuple[ResearchPaperEntity, bool]:
        """Enrich one paper and report whether its check is complete."""

        paper_url = str(paper.content.paper_url)

        try:
            html = await self.fetch_arxiv_page(
                session=session,
                paper_url=paper_url,
            )

            if not html:
                return paper, False

            repository_url = self.extract_github_repo_url(html)

            if not repository_url:
                logger.info(
                    "No explicitly linked GitHub repository: %s",
                    paper.content.title,
                )
                return paper, True

            verified, completed = await self.verify_github_repository(
                session=session,
                repository_url=repository_url,
            )

            if not completed:
                return paper, False

            if not verified:
                return paper, True

            canonical_url, stars = verified

            paper.content.github_url = HttpUrl(canonical_url)
            paper.content.github_stars = stars

            logger.info(
                "Verified repository: %s | Stars: %s",
                canonical_url,
                stars,
            )

            return paper, True

        except Exception:
            logger.exception(
                "Unexpected enrichment error for %s",
                paper_url,
            )
            return paper, False

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path,
    ) -> dict[str, ResearchPaperEntity]:
        """Load completed records from the append-only JSONL checkpoint."""

        completed = {}

        if not checkpoint_path.exists():
            return completed

        with checkpoint_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)

                    if item.get("status") != "completed":
                        continue

                    paper = ResearchPaperEntity.model_validate(
                        item["paper"]
                    )

                    paper_url = str(paper.content.paper_url)
                    completed[paper_url] = paper

                except (ValueError, TypeError, KeyError, ValidationError) as exc:
                    logger.warning(
                        "Skipping invalid checkpoint line %s: %s",
                        line_number,
                        exc,
                    )

        return completed

    @staticmethod
    def save_checkpoint(
        paper: ResearchPaperEntity,
        checkpoint_path: Path,
    ) -> None:
        """
        Append one completed record to the checkpoint.

        This avoids replacing or locking the existing checkpoint file.
        """

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        item = {
            "status": "completed",
            "paper": paper.model_dump(mode="json"),
        }

        line = json.dumps(item, ensure_ascii=False) + "\n"

        with checkpoint_path.open("a", encoding="utf-8") as file:
            file.write(line)
            file.flush()

    async def enrich(
        self,
        papers: list[ResearchPaperEntity],
        checkpoint_path: Path,
    ) -> list[ResearchPaperEntity]:
        """Resume completed work and enrich remaining papers."""

        completed = self.load_checkpoint(checkpoint_path)

        logger.info(
            "Loaded %s completed records from checkpoint.",
            len(completed),
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:

            for index, paper in enumerate(papers):
                paper_url = str(paper.content.paper_url)

                if paper_url in completed:
                    logger.info(
                        "Skipping completed paper %s/%s",
                        index + 1,
                        len(papers),
                    )
                    continue

                logger.info(
                    "Enriching paper %s/%s: %s",
                    index + 1,
                    len(papers),
                    paper.content.title,
                )

                result, is_completed = await self.enrich_one(
                    session=session,
                    paper=paper,
                )

                if is_completed:
                    self.save_checkpoint(
                        paper=result,
                        checkpoint_path=checkpoint_path,
                    )

                    completed[paper_url] = result

                else:
                    logger.warning(
                        "Not checkpointed. It will be retried "
                        "on the next run: %s",
                        paper_url,
                    )

                if index < len(papers) - 1:
                    await asyncio.sleep(ARXIV_DELAY_SECONDS)

        return [
            completed.get(
                str(paper.content.paper_url),
                paper,
            )
            for paper in papers
        ]

    @staticmethod
    def save_jsonl(
        papers: list[ResearchPaperEntity],
        output_path: Path,
    ) -> None:
        """Save final enriched entities as normal JSONL."""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = output_path.with_name(
            output_path.name + ".tmp"
        )

        with temporary_path.open("w", encoding="utf-8") as file:
            for paper in papers:
                file.write(
                    json.dumps(
                        paper.model_dump(mode="json"),
                        ensure_ascii=False,
                    ) + "\n"
                )

        # Retry replacement in case Windows temporarily locks the file.
        for attempt in range(5):
            try:
                temporary_path.replace(output_path)
                break
            except PermissionError:
                if attempt == 4:
                    logger.exception(
                        "Unable to replace final output: %s",
                        output_path,
                    )
                    raise

                time.sleep(1)

        logger.info(
            "Saved %s enriched papers to %s",
            len(papers),
            output_path,
        )


async def main(limit: int | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    input_path = Path("data/raw/arxiv_papers.jsonl")
    output_path = Path("data/processed/arxiv_papers_enriched.jsonl")
    checkpoint_path = Path(
        "data/processed/arxiv_enrichment_checkpoint.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    papers = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                paper = ResearchPaperEntity.model_validate(data)
                papers.append(paper)

            except (ValueError, TypeError, ValidationError) as exc:
                logger.warning(
                    "Skipping invalid input line %s: %s",
                    line_number,
                    exc,
                )

    if limit is not None:
        papers = papers[:limit]

    logger.info("Loaded %s research papers.", len(papers))

    enricher = GitHubEnricher()

    enriched_papers = await enricher.enrich(
        papers=papers,
        checkpoint_path=checkpoint_path,
    )

    enricher.save_jsonl(
        papers=enriched_papers,
        output_path=output_path,
    )

    completed = enricher.load_checkpoint(checkpoint_path)

    linked = sum(
        paper.content.github_url is not None
        for paper in enriched_papers
    )

    completed_in_input = sum(
        str(paper.content.paper_url) in completed
        for paper in papers
    )

    print("\n========== ENRICHMENT SUMMARY ==========")
    print(f"Input papers          : {len(papers)}")
    print(f"Completed checks      : {completed_in_input}")
    print(f"Verified GitHub links : {linked}")
    print(f"Not yet completed     : {len(papers) - completed_in_input}")
    print(f"Output file           : {output_path}")
    print(f"Checkpoint file       : {checkpoint_path}")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich arXiv papers with verified GitHub repositories."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to process.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    asyncio.run(main(limit=args.limit))
