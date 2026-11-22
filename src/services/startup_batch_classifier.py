import json
import logging
import time
from pathlib import Path

from src.config import get_settings
from src.services.gemini_classifier import (
    DailyQuotaExceeded,
    classify_startups_batch,
)


logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/raw/yc_startup_evidence.jsonl")
CHECKPOINT_FILE = Path(
    "data/processed/startup_classification_v2_checkpoint.jsonl"
)
FAILED_FILE = Path(
    "data/processed/startup_classification_v2_failed.jsonl"
)

PROMPT_VERSION = "2.0"
DEFAULT_LIMIT = 100
BATCH_SIZE = 5

# 5 seconds between requests = at most 12 requests/minute.
REQUEST_INTERVAL_SECONDS = 5


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSON at %s:%s",
                    path,
                    line_number,
                )
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def get_completed_urls() -> set[str]:
    completed = set()
    for record in read_jsonl(CHECKPOINT_FILE):
        if (
            record.get("status") == "completed"
            and record.get("prompt_version") == PROMPT_VERSION
            and record.get("source_url")
        ):
            completed.add(record["source_url"])
    return completed


def get_statistics() -> dict:
    checkpoint_records = read_jsonl(CHECKPOINT_FILE)
    completed = [
        record
        for record in checkpoint_records
        if (
            record.get("status") == "completed"
            and record.get("prompt_version") == PROMPT_VERSION
        )
    ]
    failed = [
        record
        for record in read_jsonl(FAILED_FILE)
        if record.get("prompt_version") == PROMPT_VERSION
    ]

    categories = {}
    strengths = {}
    for record in completed:
        result = record.get("result", {})
        category = result.get("classification", "UNKNOWN")
        strength = result.get("evidence_strength", "UNKNOWN")
        categories[category] = categories.get(category, 0) + 1
        strengths[strength] = strengths.get(strength, 0) + 1

    return {
        "completed": len(completed),
        "failed_records": len(failed),
        "categories": categories,
        "evidence_strengths": strengths,
    }


def classify_batch(limit: int | None = DEFAULT_LIMIT) -> None:
    settings = get_settings()
    companies = read_jsonl(INPUT_FILE)

    if not companies:
        raise RuntimeError(f"No startup evidence found: {INPUT_FILE}")

    completed_urls = get_completed_urls()
    pending = [
        company
        for company in companies
        if (
            company.get("sourceUrl")
            and company["sourceUrl"] not in completed_urls
        )
    ]

    selected = pending if limit is None else pending[:limit]
    batches = [
        selected[index:index + BATCH_SIZE]
        for index in range(0, len(selected), BATCH_SIZE)
    ]

    print("\n========== STARTUP CLASSIFICATION ==========")
    print(f"Input records        : {len(companies)}")
    print(f"Already completed    : {len(completed_urls)}")
    print(f"Pending records      : {len(pending)}")
    print(f"Selected this run    : {len(selected)}")
    print(f"API requests         : {len(batches)}")
    print(f"Companies per request: {BATCH_SIZE}")
    print(f"Gemini model         : {settings.gemini_model}")
    print("============================================\n")

    success_count = 0
    failure_count = 0
    quota_exhausted = False

    for batch_index, company_batch in enumerate(batches, start=1):
        names = [
            company.get("entityName", "Unknown")
            for company in company_batch
        ]
        print(
            f"\nRequest {batch_index}/{len(batches)}: "
            f"{', '.join(names)}"
        )

        try:
            results = classify_startups_batch(company_batch)

            # Write each validated result immediately. If the process stops
            # later, these completed records will be skipped on the next run.
            for result in results:
                record = {
                    "status": "completed",
                    "prompt_version": PROMPT_VERSION,
                    "model_name": settings.gemini_model,
                    "source_url": result.source_url,
                    "result": result.model_dump(mode="json"),
                }
                append_jsonl(CHECKPOINT_FILE, record)
                success_count += 1

                print(
                    f"  SUCCESS | {result.entity_name} | "
                    f"{result.classification.value} | "
                    f"{result.evidence_strength.value} | "
                    f"{result.confidence:.2f}"
                )

        except DailyQuotaExceeded as exc:
            # A daily quota reset is required. Do not mark these companies
            # as failed; they remain pending and will be retried next run.
            quota_exhausted = True
            print("\nDAILY GEMINI QUOTA EXHAUSTED.")
            print(f"Reason: {exc}")
            print(
                "Stopping now. This batch and all later batches remain "
                "pending; completed checkpoint records are preserved."
            )
            break

        except Exception as exc:
            logger.exception("Batch %s failed", batch_index)
            failure_count += len(company_batch)

            for company in company_batch:
                append_jsonl(
                    FAILED_FILE,
                    {
                        "status": "failed",
                        "prompt_version": PROMPT_VERSION,
                        "source_url": company.get("sourceUrl"),
                        "entity_name": company.get("entityName"),
                        "error": str(exc),
                    },
                )
                print(
                    f"  FAILED | {company.get('entityName')} | {exc}"
                )

        if batch_index < len(batches):
            time.sleep(REQUEST_INTERVAL_SECONDS)

    stats = get_statistics()
    completed_urls_after_run = get_completed_urls()
    remaining = sum(
        1
        for company in companies
        if (
            company.get("sourceUrl")
            and company["sourceUrl"] not in completed_urls_after_run
        )
    )

    print("\n========== CLASSIFICATION SUMMARY ==========")
    print(f"Successful this run  : {success_count}")
    print(f"Failed this run      : {failure_count}")
    print(f"Total completed      : {stats['completed']}")
    print(f"Failed records logged: {stats['failed_records']}")
    print(f"Still pending        : {remaining}")
    print(f"Categories           : {stats['categories']}")
    print(f"Evidence strengths   : {stats['evidence_strengths']}")
    if quota_exhausted:
        print("Status               : STOPPED - DAILY QUOTA EXHAUSTED")
    else:
        print("Status               : RUN FINISHED")
    print("============================================")


if __name__ == "__main__":
    classify_batch(limit=100)
