
import json
import os
import random
import time
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError
from pydantic import ValidationError

from src.schemas.startup_classification import StartupClassification


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_FILE = ROOT / "data/raw/yc_startup_candidates.jsonl"
EVIDENCE_FILE = ROOT / "data/raw/yc_startup_evidence.jsonl"

CHECKPOINT_FILE = (
    ROOT / "data/processed/startup_classification_v2_checkpoint.jsonl"
)

FAILED_FILE = (
    ROOT / "data/processed/startup_classification_groq_failed.jsonl"
)

BATCH_SIZE = 10
MAX_RETRIES = 2
REQUEST_DELAY = 4.0

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

VALID_CLASSIFICATIONS = {
    "AI_RELEVANT",
    "NOT_AI_RELEVANT",
    "NEEDS_REVIEW",
}


# ============================================================
# JSONL HELPERS
# ============================================================

def read_jsonl(path):
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
                print(f"Skipping invalid JSON: {path.name}, line {line_number}")

    return records


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# YC CANDIDATE FIELD EXTRACTION
# ============================================================

def get_name(record):
    content = record.get("content") or {}

    return str(
        record.get("entity_name")
        or record.get("name")
        or record.get("company_name")
        or content.get("entityName")
        or ""
    ).strip()


def get_url(record):
    source = record.get("source") or {}

    return str(
        record.get("source_url")
        or record.get("url")
        or record.get("website")
        or source.get("url")
        or ""
    ).strip()


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def load_evidence():
    """Load YC evidence using its actual flat camelCase schema."""
    evidence_map = {}
    for item in read_jsonl(EVIDENCE_FILE):
        url = (
            item.get("sourceUrl")
            or item.get("source_url")
            or item.get("url")
            or (item.get("source") or {}).get("url")
        )
        if url:
            evidence_map[url.rstrip("/")] = item
    return evidence_map


def extract_evidence(candidate, evidence_record):
    candidate_content = candidate.get("content") or {}
    candidate_data = candidate_content.get("data") or {}

    description = (
        evidence_record.get("description")
        or evidence_record.get("long_description")
        or candidate_data.get("description")
        or ""
    )
    one_liner = (
        evidence_record.get("oneLiner")
        or evidence_record.get("one_liner")
        or candidate_data.get("oneLiner")
        or candidate_data.get("one_liner")
        or ""
    )
    tags = evidence_record.get("tags") or candidate_data.get("tags") or []
    industries = evidence_record.get("industries") or []
    subindustry = evidence_record.get("subindustry") or ""

    if isinstance(tags, str):
        tags = [tags]
    if isinstance(industries, str):
        industries = [industries]

    return {
        "description": str(description)[:3500],
        "one_liner": str(one_liner)[:1200],
        "tags": tags[:25] if isinstance(tags, list) else [],
        "industries": industries[:15] if isinstance(industries, list) else [],
        "subindustry": str(subindustry)[:500],
        "evidence_found": bool(description or one_liner or tags or industries or subindustry),
    }


def get_valid_result(record):
    """
    Supports both:
    1. Existing Gemini checkpoint envelope:
       {"status": "completed", "source_url": "...", "result": {...}}

    2. Direct classification records:
       {"entity_name": "...", "source_url": "...", ...}
    """

    if record.get("status") == "completed":
        result = record.get("result") or {}
    else:
        result = record

    if not isinstance(result, dict):
        return None

    classification = result.get("classification")
    name = result.get("entity_name")
    url = result.get("source_url") or record.get("source_url")

    if classification not in VALID_CLASSIFICATIONS:
        return None

    if not name or not url:
        return None

    try:
        validated = StartupClassification(
            entity_name=name,
            source_url=url,
            classification=classification,
            evidence_strength=result["evidence_strength"],
            reasoning=result["reasoning"],
            evidence=result.get("evidence", []),
            confidence=result["confidence"],
        )

        return validated.model_dump(mode="json")

    except (ValidationError, KeyError, TypeError, ValueError):
        return None


def get_latest_checkpoint_results():
    """Latest valid checkpoint entry wins for each source URL."""
    latest = {}
    for record in read_jsonl(CHECKPOINT_FILE):
        result = get_valid_result(record)
        if result:
            url = result["source_url"].rstrip("/")
            latest[url] = {
                "result": result,
                "model_name": record.get("model_name", ""),
                "prompt_version": record.get("prompt_version", ""),
            }
    return latest


def get_completed_urls():
    return set(get_latest_checkpoint_results())


def make_prompt(batch):
    companies = []

    for index, company in enumerate(batch, start=1):
        companies.append({
            "index": index,
            "name": company["name"],
            "description": company["description"],
            "one_liner": company["one_liner"],
            "tags": company["tags"],
            "industries": company["industries"],
            "subindustry": company["subindustry"],
        })

    return f"""
You are a conservative AI startup classification analyst.

Classify every company using ONLY its supplied evidence.

Return exactly one valid JSON object in this format:

{{
  "classifications": [
    {{
      "index": 1,
      "classification": "AI_RELEVANT",
      "evidence_strength": "STRONG",
      "reasoning": "Short explanation based on supplied evidence.",
      "evidence": ["Specific evidence from the supplied information"],
      "confidence": 0.95
    }}
  ]
}}

Allowed classification:
- AI_RELEVANT
- NOT_AI_RELEVANT
- NEEDS_REVIEW

Allowed evidence_strength:
- STRONG
- MODERATE
- WEAK

Classification rules:

1. AI_RELEVANT:
   AI/ML is clearly central to the company's product, service,
   infrastructure, research, or core technology.

2. NOT_AI_RELEVANT:
   Available evidence clearly establishes that the core offering
   is not AI/ML-related.

3. NEEDS_REVIEW:
   Evidence is ambiguous, insufficient, or does not establish
   whether AI/ML is central.

4. Generic SaaS, analytics, automation, software, or technology
   is not sufficient evidence of AI relevance.

5. If the supplied description or one-liner explicitly identifies
   machine learning, AI models, LLMs, generative AI, computer vision,
   speech AI, or AI infrastructure as part of the offering, classify
   AI_RELEVANT unless evidence clearly contradicts this.

6. Do not use NEEDS_REVIEW merely because evidence is brief. Use it
   only when the available facts genuinely do not establish relevance.

7. Do not infer AI capabilities from a company name or domain.

8. Never invent products, features, customers, funding, or technologies.

9. Return exactly one classification for every input index.

10. Do not return company names or URLs. The program will associate
    each result with its original input record.

11. Output JSON only. Do not use Markdown code fences.

INPUT COMPANIES:

{json.dumps(companies, ensure_ascii=False)}
"""


# ============================================================
# GROQ REQUEST
# ============================================================

def classify_batch(client, batch):
    prompt = make_prompt(batch)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise classification engine. "
                            "Return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Groq returned an empty response")

            parsed = json.loads(content)

            results = parsed.get("classifications")

            if not isinstance(results, list):
                raise ValueError(
                    "Response does not contain a classifications array"
                )

            return results

        except RateLimitError as exc:
            if attempt >= MAX_RETRIES:
                raise

            # Do not let the SDK sleep automatically.
            wait = min(30, 5 * (2 ** attempt)) + random.uniform(1, 3)

            print(
                f"Groq rate limited the request. "
                f"Retrying in {wait:.1f} seconds."
            )

            time.sleep(wait)

        except APIStatusError as exc:
            if exc.status_code == 429:
                # Stop instead of endlessly retrying.
                raise

            if exc.status_code not in (500, 502, 503, 504):
                raise

            if attempt >= MAX_RETRIES:
                raise

            wait = min(20, 3 * (2 ** attempt))
            print(f"Groq server error. Retrying in {wait} seconds.")
            time.sleep(wait)

    return []


# ============================================================
# VALIDATE AND SAVE RESULTS
# ============================================================

def validate_and_save(batch, results):
    saved = 0
    seen_indices = set()

    for item in results:
        if not isinstance(item, dict):
            continue

        index = None
        try:
            index = int(item.get("index"))

            if index < 1 or index > len(batch):
                continue

            if index in seen_indices:
                continue

            seen_indices.add(index)

            original = batch[index - 1]

            classification = StartupClassification(
                entity_name=original["name"],
                source_url=original["url"],
                classification=item["classification"],
                evidence_strength=item["evidence_strength"],
                reasoning=item["reasoning"],
                evidence=item.get("evidence", []),
                confidence=item["confidence"],
            )

            result = classification.model_dump(mode="json")

            # Preserve the existing checkpoint envelope format.
            checkpoint_record = {
                "status": "completed",
                "prompt_version": "3.0",
                "model_name": MODEL,
                "source_url": original["url"],
                "result": result,
            }

            append_jsonl(CHECKPOINT_FILE, checkpoint_record)

            print(
                f"  Saved: {original['name']} -> "
                f"{result['classification']}"
            )

            saved += 1

        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            append_jsonl(FAILED_FILE, {
                "name": batch[index - 1]["name"]
                if isinstance(index, int) and 1 <= index <= len(batch)
                else None,
                "error": str(exc),
                "model_result": item,
            })

    return saved


# ============================================================
# MAIN
# ============================================================

def main():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Check your .env file."
        )

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(CANDIDATES_FILE)

    candidates = read_jsonl(CANDIDATES_FILE)
    evidence_map = load_evidence()

    latest_results = get_latest_checkpoint_results()
    completed_urls = set(latest_results)

    pending = []
    seen_candidate_urls = set()

    for candidate in candidates:
        name = get_name(candidate)
        url = get_url(candidate)

        if not name or not url:
            continue

        if url in seen_candidate_urls:
            continue

        seen_candidate_urls.add(url)

        normalized_url = url.rstrip("/")
        previous = latest_results.get(normalized_url)

        # Reclassify old Groq rows: the previous loader missed YC's sourceUrl key.
        if previous and previous.get("model_name") != MODEL:
            continue

        evidence_record = evidence_map.get(normalized_url, {})

        evidence = extract_evidence(
            candidate,
            evidence_record,
        )

        pending.append({
            "name": name,
            "url": url,
            "description": evidence["description"],
            "one_liner": evidence["one_liner"],
            "tags": evidence["tags"],
            "industries": evidence["industries"],
            "subindustry": evidence["subindustry"],
            "evidence_found": evidence["evidence_found"],
        })

    total_candidates = len(seen_candidate_urls)
    already_classified = len(completed_urls)

    print("\n========== STARTUP CLASSIFICATION ==========")
    print(f"Total unique candidates: {total_candidates}")
    print(f"Already classified: {already_classified}")
    print(f"Remaining: {len(pending)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model: {MODEL}")
    print("============================================\n")

    if not pending:
        print("No pending startups.")
        return

    client = Groq(
        api_key=api_key,
        max_retries=0,
        timeout=60.0,
    )

    total_saved = 0
    total_batches = (
        len(pending) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    try:
        for batch_number, start in enumerate(
            range(0, len(pending), BATCH_SIZE),
            start=1,
        ):
            batch = pending[start:start + BATCH_SIZE]

            missing_evidence = [r["name"] for r in batch if not r["evidence_found"]]
            if missing_evidence:
                print("Missing YC evidence; skipping batch:", ", ".join(missing_evidence))
                continue

            print(
                f"\nBatch {batch_number}/{total_batches} "
                f"({len(batch)} startups)"
            )

            try:
                results = classify_batch(client, batch)

                saved = validate_and_save(batch, results)
                total_saved += saved

                print(f"Batch completed: {saved}/{len(batch)}")

                current_completed = len(get_completed_urls())

                print(
                    f"Total completed: {current_completed}"
                )

                if saved < len(batch):
                    print(
                        "WARNING: Some results were missing or invalid. "
                        "They will remain pending for the next run."
                    )

            except RateLimitError as exc:
                print("\nGroq rate limit reached.")
                print("Stopping safely. Existing checkpoint is preserved.")
                print(f"Details: {exc}")
                break

            except APIStatusError as exc:
                if exc.status_code == 429:
                    print("\nGroq HTTP 429 limit reached.")
                    print("Stopping safely. Rerun when the limit resets.")
                    break

                print(
                    f"Batch API error: {exc.status_code} - {exc}"
                )

                append_jsonl(FAILED_FILE, {
                    "batch_number": batch_number,
                    "urls": [record["url"] for record in batch],
                    "error": str(exc),
                })

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                print("All successfully saved checkpoint records are safe.")
                break

            except Exception as exc:
                print(
                    f"Batch error: {type(exc).__name__}: {exc}"
                )

                append_jsonl(FAILED_FILE, {
                    "batch_number": batch_number,
                    "urls": [record["url"] for record in batch],
                    "error": str(exc),
                })

            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print("\nStopped safely.")

    finally:
        client.close()

    final_completed = len(get_completed_urls())

    final_counts = Counter()
    for item in get_latest_checkpoint_results().values():
        final_counts[item["result"]["classification"]] += 1

    print("\n========== FINAL SUMMARY ==========")
    print(f"Total unique candidates: {total_candidates}")
    print(f"Valid classifications: {final_completed}")
    print(f"New classifications this run: {total_saved}")
    print(f"Remaining: {max(0, total_candidates - final_completed)}")
    print(f"Classification counts: {dict(final_counts)}")
    print("===================================\n")


if __name__ == "__main__":
    main()
