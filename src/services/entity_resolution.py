
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[2]

STARTUP_CANDIDATES = ROOT / "data/raw/yc_startup_candidates.jsonl"
STARTUP_CLASSIFICATIONS = ROOT / "data/processed/startup_classification_v2_checkpoint.jsonl"
PRODUCTS = ROOT / "data/processed/products.jsonl"

OUTPUT_DIR = ROOT / "data/processed"
SEED_OUTPUT = OUTPUT_DIR / "canonical_seed_entities.jsonl"
MAPPING_OUTPUT = OUTPUT_DIR / "entity_mapping_log.jsonl"
STARTUP_OUTPUT = OUTPUT_DIR / "startups_resolved.jsonl"
PRODUCT_OUTPUT = OUTPUT_DIR / "products_resolved.jsonl"

SEED_LIMIT = 50
FUZZY_THRESHOLD = 97


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON: {path.name}, line {line_number}: {exc}")

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_name(value: str | None) -> str:
    if not value:
        return ""

    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()

    value = re.sub(
        r"\b(incorporated|corporation|limited|private limited|pvt ltd|"
        r"ltd|inc|llc|llp|technologies|technology|technologies)\b",
        " ",
        value,
    )

    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def get_startup_name(record: dict[str, Any]) -> str:
    return (
        record.get("content", {})
        .get("entityName", "")
        .strip()
    )


def get_product_name(record: dict[str, Any]) -> str:
    return (
        record.get("content", {})
        .get("productName", "")
        .strip()
    )


def get_source_url(record: dict[str, Any]) -> str:
    return (
        record.get("source", {}).get("url")
        or record.get("source_url")
        or record.get("sourceUrl")
        or ""
    )


def build_startup_seeds(
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible = []

    for record in classifications:
        if record.get("status") != "completed":
            continue

        result = record.get("result") or {}

        if result.get("classification") != "AI_RELEVANT":
            continue

        name = result.get("entity_name", "").strip()
        if not name:
            continue

        strength = result.get("evidence_strength", "WEAK")
        confidence = float(result.get("confidence", 0))

        eligible.append({
            "name": name,
            "normalized_name": normalize_name(name),
            "entity_type": "STARTUP",
            "source_url": result.get("source_url", ""),
            "evidence_strength": strength,
            "confidence": confidence,
        })

    eligible.sort(
        key=lambda item: (
            item["evidence_strength"] != "STRONG",
            -item["confidence"],
            item["normalized_name"],
        )
    )

    seeds = []
    seen = set()

    for item in eligible:
        normalized = item["normalized_name"]

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)

        item["canonical_id"] = f"STARTUP-{len(seeds) + 1:03d}"
        seeds.append(item)

        if len(seeds) >= SEED_LIMIT:
            break

    return seeds


def build_product_seeds(
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible = []

    for record in products:
        name = get_product_name(record)
        normalized = normalize_name(name)

        if not name or not normalized:
            continue

        eligible.append({
            "name": name,
            "normalized_name": normalized,
            "entity_type": "PRODUCT",
            "source_url": get_source_url(record),
        })

    eligible.sort(key=lambda item: item["normalized_name"])

    seeds = []
    seen = set()

    for item in eligible:
        normalized = item["normalized_name"]

        if normalized in seen:
            continue

        seen.add(normalized)

        item["canonical_id"] = f"PRODUCT-{len(seeds) + 1:03d}"
        seeds.append(item)

        if len(seeds) >= SEED_LIMIT:
            break

    return seeds


def resolve_name(
    name: str,
    seeds: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized = normalize_name(name)

    if not normalized:
        return {
            "matched": False,
            "canonical_id": None,
            "canonical_name": None,
            "match_method": "NO_VALID_NAME",
            "match_score": 0,
        }

    # First pass: exact normalized name.
    for seed in seeds:
        if normalized == seed["normalized_name"]:
            return {
                "matched": True,
                "canonical_id": seed["canonical_id"],
                "canonical_name": seed["name"],
                "match_method": "EXACT_NORMALIZED",
                "match_score": 100,
            }

    # Second pass: conservative fuzzy matching.
    if len(normalized) < 6:
        return {
            "matched": False,
            "canonical_id": None,
            "canonical_name": None,
            "match_method": "NO_MATCH",
            "match_score": 0,
        }

    best_seed = None
    best_score = 0

    for seed in seeds:
        score = fuzz.ratio(normalized, seed["normalized_name"])

        if score > best_score:
            best_score = score
            best_seed = seed

    if best_seed and best_score >= FUZZY_THRESHOLD:
        return {
            "matched": True,
            "canonical_id": best_seed["canonical_id"],
            "canonical_name": best_seed["name"],
            "match_method": "FUZZY_HIGH_CONFIDENCE",
            "match_score": round(best_score, 2),
        }

    return {
        "matched": False,
        "canonical_id": None,
        "canonical_name": None,
        "match_method": "NO_MATCH",
        "match_score": round(best_score, 2),
    }


def resolve_records(
    records: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    entity_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = []
    mapping_log = []

    for record in records:
        if entity_type == "STARTUP":
            name = get_startup_name(record)
        else:
            name = get_product_name(record)

        if not name:
            continue

        match = resolve_name(name, seeds)
        source_url = get_source_url(record)

        resolution = {
            "entity_type": entity_type,
            "original_name": name,
            **match,
        }

        output_record = dict(record)
        output_record["entityResolution"] = resolution
        resolved.append(output_record)

        mapping_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entity_type": entity_type,
            "source_name": name,
            "source_url": source_url,
            "canonical_id": match["canonical_id"],
            "canonical_name": match["canonical_name"],
            "match_method": match["match_method"],
            "match_score": match["match_score"],
            "status": "MATCHED" if match["matched"] else "UNMATCHED",
        })

    return resolved, mapping_log


def main() -> None:
    print("Loading source records...")

    startup_records = read_jsonl(STARTUP_CANDIDATES)
    classifications = read_jsonl(STARTUP_CLASSIFICATIONS)
    product_records = read_jsonl(PRODUCTS)

    print(f"Startup candidates: {len(startup_records)}")
    print(f"Classification records: {len(classifications)}")
    print(f"Products: {len(product_records)}")

    print("\nBuilding canonical seed entities...")

    startup_seeds = build_startup_seeds(classifications)
    product_seeds = build_product_seeds(product_records)

    all_seeds = startup_seeds + product_seeds
    write_jsonl(SEED_OUTPUT, all_seeds)

    print(f"Canonical startup seeds: {len(startup_seeds)}")
    print(f"Canonical product seeds: {len(product_seeds)}")
    print(f"Total canonical seeds: {len(all_seeds)}")

    print("\nResolving startup names...")

    resolved_startups, startup_log = resolve_records(
        startup_records,
        startup_seeds,
        "STARTUP",
    )

    print("Resolving product names...")

    resolved_products, product_log = resolve_records(
        product_records,
        product_seeds,
        "PRODUCT",
    )

    combined_log = startup_log + product_log

    write_jsonl(STARTUP_OUTPUT, resolved_startups)
    write_jsonl(PRODUCT_OUTPUT, resolved_products)
    write_jsonl(MAPPING_OUTPUT, combined_log)

    matched = sum(
        1 for record in combined_log
        if record["status"] == "MATCHED"
    )
    unmatched = len(combined_log) - matched

    print("\n========== ENTITY RESOLUTION REPORT ==========")
    print(f"Startup seeds: {len(startup_seeds)}")
    print(f"Product seeds: {len(product_seeds)}")
    print(f"Startup records resolved: {len(resolved_startups)}")
    print(f"Product records resolved: {len(resolved_products)}")
    print(f"Total mapping log entries: {len(combined_log)}")
    print(f"Matched: {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Seed file: {SEED_OUTPUT.relative_to(ROOT)}")
    print(f"Mapping log: {MAPPING_OUTPUT.relative_to(ROOT)}")
    print(f"Resolved startups: {STARTUP_OUTPUT.relative_to(ROOT)}")
    print(f"Resolved products: {PRODUCT_OUTPUT.relative_to(ROOT)}")
    print("===============================================")


if __name__ == "__main__":
    main()
