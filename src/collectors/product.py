import asyncio
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import HttpUrl

from src.schemas.entities import (
    ProductContent,
    ProductEntity,
    PricingModel,
    SourceInfo,
)

logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/raw/everyai_tools_raw.json")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "products.jsonl"

SOURCE_NAME = "EveryAI"

# Specific AI technologies and product capabilities.
AI_PATTERNS = [
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\blarge language models?\b",
    r"\bllms?\b",
    r"\bgenerative ai\b",
    r"\bchatgpt\b",
    r"\bgpt[-\s]?\d",
    r"\bopenai\b",
    r"\bclaude\b",
    r"\bgemini\b",
    r"\bcopilot\b",
    r"\bcomputer vision\b",
    r"\btext[- ]to[- ]image\b",
    r"\btext[- ]to[- ]speech\b",
    r"\bspeech[- ]to[- ]text\b",
    r"\bstable diffusion\b",
    r"\bneural networks?\b",
    r"\bai[- ]powered\b",
    r"\bai[- ]driven\b",
    r"\bai[- ]generated\b",
    r"\bai[- ]assistant\b",
    r"\bai[- ]agent\b",
    r"\bai[- ]model\b",
    r"\bai[- ]image\b",
    r"\bai[- ]video\b",
    r"\bai[- ]writing\b",
    r"\bai[- ]coding\b",
    r"\bai[- ]voice\b",
    r"\bai[- ]music\b",
    r"\bai[- ]art\b",
]

AI_REGEX = re.compile("|".join(AI_PATTERNS), re.IGNORECASE)

# Generic phrases alone do not establish that a product is AI-powered.
WEAK_ONLY = re.compile(
    r"\b(ai template|ai templates|ai prompts?|mega prompts?|"
    r"chatgpt prompts?|prompt packs?)\b",
    re.IGNORECASE,
)


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(str(item) for item in value)

    return str(value).strip().strip('"').strip("'")


def normalize_name(value):
    value = unicodedata.normalize("NFKC", clean_text(value))
    value = re.sub(r"\s+", " ", value)
    return value.casefold().strip()


def normalize_url(value):
    value = clean_text(value)

    if not value:
        return None

    try:
        parts = urlsplit(value)

        if parts.scheme not in ("http", "https") or not parts.netloc:
            return None

        # Remove tracking query parameters and fragments.
        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/") or "/",
            "",
            "",
        ))

    except ValueError:
        return None


def is_ai_product(record):
    """
    Conservative evidence filter.

    Requires AI-specific evidence in the product name,
    description, category, use case, or features.
    """

    fields = [
        clean_text(record.get("name")),
        clean_text(record.get("description")),
        clean_text(record.get("category")),
        clean_text(record.get("use_case")),
        clean_text(record.get("features")),
    ]

    evidence_text = " ".join(fields)

    if not AI_REGEX.search(evidence_text):
        return False

    # Prompt collections alone are weak evidence.
    if (
        WEAK_ONLY.search(evidence_text)
        and not re.search(
            r"\b(machine learning|llm|large language model|"
            r"generative ai|ai[- ]powered|ai[- ]generated|"
            r"neural network|computer vision)\b",
            evidence_text,
            re.IGNORECASE,
        )
    ):
        return False

    return True


def build_product(record):
    name = clean_text(record.get("name"))
    listing_url = normalize_url(record.get("everyai_url"))
    official_url = normalize_url(record.get("url"))

    if not name or not listing_url:
        return None

    description = clean_text(record.get("description"))
    category = clean_text(record.get("category"))
    use_case = clean_text(record.get("use_case"))

    features = record.get("features") or []

    if not isinstance(features, list):
        features = [clean_text(features)]

    features = [
        clean_text(item)
        for item in features
        if clean_text(item)
    ]

    # Pricing and startup association are not present in this feed.
    # Do not invent either value.
    product_data = {
        "recordType": "PRODUCT",
        "schemaVersion": "1.0",
        "source": {
            "name": SOURCE_NAME,
            "url": listing_url,
        },
        "content": {
            "productName": name,
            "startupName": None,
            "pricingModel": None,
            "category": category or None,
            "description": description or None,
            "officialUrl": official_url,
            "useCase": use_case or None,
            "features": features,
            "added": clean_text(record.get("added")) or None,
        },
        "collectedAt": datetime.now(timezone.utc).isoformat(),
    }

    return product_data


async def collect_products():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("EveryAI input must contain a JSON list.")

    seen_names = set()
    seen_urls = set()

    output = []
    rejected = 0
    duplicates = 0

    for record in records:
        if not isinstance(record, dict):
            rejected += 1
            continue

        if not is_ai_product(record):
            rejected += 1
            continue

        product = build_product(record)

        if product is None:
            rejected += 1
            continue

        name_key = normalize_name(
            product["content"]["productName"]
        )

        url_key = product["source"]["url"]

        if name_key in seen_names or url_key in seen_urls:
            duplicates += 1
            continue

        seen_names.add(name_key)
        seen_urls.add(url_key)

        output.append(product)

    # Write complete output only after processing.
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for product in output:
            file.write(
                json.dumps(product, ensure_ascii=False) + "\n"
            )

    print("\n========== PRODUCT COLLECTION REPORT ==========")
    print(f"Source records:       {len(records)}")
    print(f"Accepted products:    {len(output)}")
    print(f"Rejected records:     {rejected}")
    print(f"Duplicate records:    {duplicates}")
    print(f"Unique product names: {len(seen_names)}")
    print(f"Unique source URLs:   {len(seen_urls)}")
    print(f"Output:               {OUTPUT_FILE}")

    if len(output) >= 1000:
        print("\nTARGET MET: At least 1,000 candidate products collected.")
    else:
        print("\nTARGET NOT MET: Fewer than 1,000 products collected.")

    print("===============================================\n")


if __name__ == "__main__":
    asyncio.run(collect_products())