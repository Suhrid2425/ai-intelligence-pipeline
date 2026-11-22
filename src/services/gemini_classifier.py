import json
import logging
import random
import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import ValidationError

from src.config import get_settings
from src.schemas.startup_classification import StartupClassification


logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_BATCH_SIZE = 5

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


class DailyQuotaExceeded(RuntimeError):
    """Raised when the Gemini free-tier daily request quota is exhausted."""


def _is_daily_quota_error(exception: APIError) -> bool:
    """
    Distinguish a daily quota exhaustion from a temporary per-minute 429.

    Gemini's error body may contain a quota ID, a human-readable message,
    or both. Check all of them rather than relying on one exact message.
    """
    message = str(exception).lower()

    daily_quota_markers = (
        "per day",
        "perday",
        "requestsperday",
        "generate_content_free_tier_requests",
        "daily quota",
        "per_day",
    )

    return any(marker in message for marker in daily_quota_markers)


def _should_retry(exception: APIError) -> bool:
    """Return whether an API error is normally transient."""
    return getattr(exception, "code", None) in RETRYABLE_STATUS_CODES


def _validate_company_input(company: dict) -> tuple[str, str]:
    """Validate and return the company's exact name and source URL."""
    entity_name = company.get("entityName")
    source_url = company.get("sourceUrl")

    if not isinstance(entity_name, str) or not entity_name.strip():
        raise ValueError("A valid startup name is required.")

    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("A valid startup source URL is required.")

    return entity_name.strip(), source_url.strip()


def build_classification_prompt(company: dict) -> str:
    """Build the strict evidence-based prompt for one startup."""
    evidence = {
        "name": company.get("entityName"),
        "website": company.get("sourceUrl"),
        "one_liner": company.get("oneLiner"),
        "description": company.get("description"),
        "tags": company.get("tags", []),
        "industries": company.get("industries", []),
        "subindustry": company.get("subindustry"),
    }

    return f"""
You are an expert AI industry research analyst.

Determine whether this company is genuinely AI-relevant.

COMPANY EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

CLASSIFICATION DEFINITIONS:

AI_RELEVANT:
Choose this when evidence supports that:
- The company builds AI/ML models, infrastructure or tools.
- The company provides a product with demonstrable AI-powered
  functionality.
- AI/ML is a meaningful part of its commercial offering.

NOT_AI_RELEVANT:
Choose this when available evidence clearly establishes that:
- AI/ML is not a meaningful part of its offering.
- It provides ordinary software, marketplaces or SaaS
  without demonstrated AI functionality.

NEEDS_REVIEW:
Choose this when:
- Evidence is insufficient or contradictory.
- AI relevance is suggested only by a tag.
- The description mentions automation, analytics or prediction
  without establishing an AI/ML component.
- A one-liner and longer description materially disagree.
- The company's actual product cannot be determined reliably.

EVIDENCE STRENGTH:

STRONG:
Explicit evidence of AI/ML technology or a demonstrably
AI-powered product.

MODERATE:
Specific AI-related functionality is indicated, but its
technical implementation or role is not fully explained.

WEAK:
Only indirect, generic, ambiguous or tag-based evidence exists.

IMPORTANT RULES:
1. Give greater weight to explicit product descriptions than tags.
2. Never infer AI capability from a company name.
3. Do not assume predictive analytics always means ML.
4. Do not assume image uploads automatically mean computer vision.
5. Do not assume fraud detection automatically means ML.
6. Do not assume a company builds AI merely because it uses AI.
7. Do not invent products, technologies or customers.
8. Use only the supplied evidence.
9. Preserve contradictions rather than inventing explanations.
10. If evidence is insufficient, choose NEEDS_REVIEW.
11. Include exact supporting details in the evidence array.
12. Preserve the exact company name and source URL.

CONFIDENCE GUIDELINES:
0.90-1.00: Strong and explicit evidence.
0.75-0.89: Good evidence with minor uncertainty.
0.50-0.74: Limited or ambiguous evidence.
Below 0.50: Very weak evidence.

Confidence represents your assessment of evidence quality.
Do not automatically assign high confidence.

Return only structured JSON matching the required schema.
"""


def build_batch_classification_prompt(companies: list[dict]) -> str:
    """Build one evidence-based prompt for a small batch of startups."""
    evidence = []

    for index, company in enumerate(companies, start=1):
        name, source_url = _validate_company_input(company)

        evidence.append(
            {
                "batch_item": index,
                "name": name,
                "website": source_url,
                "one_liner": company.get("oneLiner"),
                "description": company.get("description"),
                "tags": company.get("tags", []),
                "industries": company.get("industries", []),
                "subindustry": company.get("subindustry"),
            }
        )

    return f"""
You are an expert AI industry research analyst.

Classify every company in the supplied dataset independently.

COMPANY EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

CLASSIFICATION DEFINITIONS:

AI_RELEVANT:
- Builds AI/ML models, infrastructure or tools.
- Offers a product with demonstrable AI-powered functionality.
- AI/ML is a meaningful part of its commercial offering.

NOT_AI_RELEVANT:
- Available evidence clearly establishes that AI/ML is not meaningful
  to the company's offering.
- Ordinary software, marketplaces or SaaS without demonstrated
  AI functionality.

NEEDS_REVIEW:
- Evidence is insufficient or contradictory.
- AI relevance is suggested only by a tag.
- Automation, analytics or prediction is mentioned without
  establishing an AI/ML component.
- The actual product cannot be determined reliably.

EVIDENCE STRENGTH:
STRONG: Explicit evidence of AI/ML technology or a demonstrably
AI-powered product.
MODERATE: Specific AI-related functionality is indicated, but its
technical implementation or role is not fully explained.
WEAK: Only indirect, generic, ambiguous or tag-based evidence exists.

STRICT RULES:
1. Use only the supplied evidence.
2. Never infer AI capability from a company name.
3. Do not assume predictive analytics always means ML.
4. Do not assume image uploads mean computer vision.
5. Do not assume fraud detection automatically means ML.
6. Do not assume a company builds AI merely because it uses AI.
7. Never invent products, technologies or customers.
8. Give greater weight to explicit descriptions than tags.
9. Preserve contradictions.
10. Use NEEDS_REVIEW when evidence is insufficient.
11. Include exact supporting details in the evidence array.
12. Preserve the exact company name and source URL.
13. Return exactly one result for every supplied company.
14. Do not merge companies or omit any company.
15. Return a JSON array, not an object containing an array.

CONFIDENCE:
0.90-1.00: Strong explicit evidence.
0.75-0.89: Good evidence with minor uncertainty.
0.50-0.74: Limited or ambiguous evidence.
Below 0.50: Very weak evidence.

Return only structured JSON matching the required schema.
"""


def build_response_schema() -> types.Schema:
    """Define the Gemini-compatible structured JSON schema."""
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "entity_name": types.Schema(type=types.Type.STRING),
            "source_url": types.Schema(type=types.Type.STRING),
            "classification": types.Schema(
                type=types.Type.STRING,
                enum=[
                    "AI_RELEVANT",
                    "NOT_AI_RELEVANT",
                    "NEEDS_REVIEW",
                ],
            ),
            "evidence_strength": types.Schema(
                type=types.Type.STRING,
                enum=["STRONG", "MODERATE", "WEAK"],
            ),
            "reasoning": types.Schema(type=types.Type.STRING),
            "evidence": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
            ),
            "confidence": types.Schema(type=types.Type.NUMBER),
        },
        required=[
            "entity_name",
            "source_url",
            "classification",
            "evidence_strength",
            "reasoning",
            "evidence",
            "confidence",
        ],
    )


def _build_client():
    """Create a Gemini client using the configured API key."""
    settings = get_settings()

    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    return settings, genai.Client(api_key=settings.gemini_api_key)


def _validate_company_identity(
    result: StartupClassification,
    entity_name: str,
    source_url: str,
) -> None:
    """Ensure the returned classification belongs to the requested company."""
    if result.entity_name.strip().casefold() != entity_name.strip().casefold():
        raise ValueError(
            f"Company name mismatch: expected {entity_name}, "
            f"received {result.entity_name}"
        )

    if result.source_url.strip() != source_url.strip():
        raise ValueError(f"Source URL mismatch for {entity_name}")


def _raise_or_retry(
    exception: APIError,
    attempt: int,
    operation_name: str,
) -> None:
    """
    Raise daily quota errors immediately.

    For other retryable API errors, use short exponential backoff.
    This function never sleeps for a daily quota reset.
    """
    if _is_daily_quota_error(exception):
        raise DailyQuotaExceeded(
            "Gemini daily free-tier request quota is exhausted. "
            "Stop processing and resume after the quota resets."
        ) from exception

    if not _should_retry(exception) or attempt >= MAX_ATTEMPTS:
        raise exception

    delay = min((2 ** attempt) + random.uniform(0, 1), 15)

    logger.warning(
        "%s API error. Attempt %s/%s. Retrying in %.1f seconds.",
        operation_name,
        attempt,
        MAX_ATTEMPTS,
        delay,
    )
    time.sleep(delay)


def classify_startup(company: dict) -> StartupClassification:
    """
    Classify one startup using Gemini.

    Includes structured JSON generation, Pydantic validation,
    identity verification, and transient-error retries.
    """
    entity_name, source_url = _validate_company_input(company)
    settings, client = _build_client()

    try:
        prompt = build_classification_prompt(company)
        response_schema = build_response_schema()

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1,
                    ),
                )

                if not response.text:
                    raise ValueError(
                        f"Gemini returned an empty response for {entity_name}."
                    )

                result = StartupClassification.model_validate_json(
                    response.text
                )

                _validate_company_identity(
                    result=result,
                    entity_name=entity_name,
                    source_url=source_url,
                )

                logger.info(
                    "Classified %s as %s | Evidence: %s | Confidence: %.2f",
                    entity_name,
                    result.classification.value,
                    result.evidence_strength.value,
                    result.confidence,
                )
                return result

            except APIError as exc:
                _raise_or_retry(exc, attempt, "Gemini single-company")

            except ValidationError:
                logger.exception(
                    "Invalid classification response for %s",
                    entity_name,
                )
                raise

    finally:
        client.close()

    raise RuntimeError(
        f"Gemini classification failed for {entity_name} "
        f"after {MAX_ATTEMPTS} attempts."
    )


def classify_startups_batch(
    companies: list[dict],
) -> list[StartupClassification]:
    """
    Classify up to five startups in one Gemini request.

    Every result is validated independently and matched to its
    original company name and source URL. No partial batch is returned.
    """
    if not companies:
        return []

    if len(companies) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Maximum batch size is {MAX_BATCH_SIZE} startups."
        )

    expected: dict[str, str] = {}

    for company in companies:
        name, source_url = _validate_company_input(company)

        if source_url in expected:
            raise ValueError(f"Duplicate source URL in batch: {source_url}")

        expected[source_url] = name

    settings, client = _build_client()

    try:
        prompt = build_batch_classification_prompt(companies)
        response_schema = types.Schema(
            type=types.Type.ARRAY,
            items=build_response_schema(),
        )

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1,
                    ),
                )

                if not response.text:
                    raise ValueError("Gemini returned an empty batch response.")

                raw_results: Any = json.loads(response.text)

                if not isinstance(raw_results, list):
                    raise ValueError("Batch response must be a JSON array.")

                if len(raw_results) != len(companies):
                    raise ValueError(
                        f"Expected {len(companies)} results, "
                        f"received {len(raw_results)}."
                    )

                validated: list[StartupClassification] = []
                returned_urls: set[str] = set()

                for item in raw_results:
                    result = StartupClassification.model_validate(item)
                    result_url = result.source_url.strip()

                    if result_url in returned_urls:
                        raise ValueError(
                            f"Duplicate company returned: {result_url}"
                        )

                    if result_url not in expected:
                        raise ValueError(
                            f"Unexpected source URL returned: {result_url}"
                        )

                    returned_urls.add(result_url)

                    _validate_company_identity(
                        result=result,
                        entity_name=expected[result_url],
                        source_url=result_url,
                    )

                    validated.append(result)

                if returned_urls != set(expected):
                    raise ValueError(
                        "Batch response is missing one or more companies."
                    )

                return validated

            except APIError as exc:
                _raise_or_retry(exc, attempt, "Gemini batch")

            except (json.JSONDecodeError, ValidationError):
                logger.exception("Gemini returned invalid batch JSON.")
                raise

    finally:
        client.close()

    raise RuntimeError("Gemini batch classification failed.")
