from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "submission"
OUT.mkdir(parents=True, exist_ok=True)

# Remove characters that Excel does not support.
ILLEGAL_EXCEL_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing: {path.relative_to(ROOT)}")
        return []

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Invalid JSON {path.name}:{line_no}: {exc}")

    return rows


def flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else str(key)
            flatten(new_key, value, out)

    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)

    else:
        out[prefix] = obj


def json_cell(value: Any) -> str | int | float | bool | None:
    """Convert nested values and sanitize text for Excel and CSV."""

    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)

    if isinstance(value, str):
        value = ILLEGAL_EXCEL_CHARACTERS.sub("", value)
        return value[:32767]

    return value


def write_csv(
    name: str,
    headers: list[str],
    rows: list[dict[str, Any]]
) -> None:

    path = OUT / f"{name}.csv"

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: json_cell(row.get(key))
                for key in headers
            })


# ============================================================
# STARTUPS
# ============================================================

candidates = read_jsonl(
    RAW / "yc_startup_candidates.jsonl"
)

evidence_rows = read_jsonl(
    RAW / "yc_startup_evidence.jsonl"
)

classification_path = (
    PROCESSED / "startup_classification_v2_checkpoint.jsonl"
)

if not classification_path.exists():
    classification_path = (
        PROCESSED / "startup_classification_checkpoint.jsonl"
    )

classifications = read_jsonl(classification_path)

evidence_by_url = {
    str(item.get("sourceUrl", "")).rstrip("/").lower(): item
    for item in evidence_rows
}

classification_by_url = {}

for item in classifications:

    result = item.get("result") or {}

    url = str(
        item.get("source_url")
        or result.get("source_url")
        or ""
    ).rstrip("/").lower()

    if url:
        classification_by_url[url] = item


startup_rows = []

for item in candidates:

    source = item.get("source") or {}
    content = item.get("content") or {}
    data = content.get("data") or {}

    url = str(source.get("url", ""))
    key = url.rstrip("/").lower()

    evidence = evidence_by_url.get(key, {})
    classification = classification_by_url.get(key, {})
    result = classification.get("result") or {}

    startup_rows.append({
        "startup_name": content.get("entityName"),
        "source_name": source.get("name"),
        "source_url": url,
        "employee_count": data.get(
            "employeeCount",
            evidence.get("teamSize")
        ),
        "one_liner": evidence.get("oneLiner"),
        "description": evidence.get("description"),
        "tags": evidence.get("tags"),
        "industries": evidence.get("industries"),
        "classification_status": classification.get(
            "status",
            "pending"
        ),
        "ai_classification": result.get(
            "classification",
            "PENDING"
        ),
        "evidence_strength": result.get("evidence_strength"),
        "confidence": result.get("confidence"),
        "reasoning": result.get("reasoning"),
        "evidence": result.get("evidence"),
        "collected_at": item.get("collectedAt"),
        "model_name": classification.get("model_name"),
        "prompt_version": classification.get("prompt_version"),
    })


# ============================================================
# PRODUCTS
# ============================================================

product_input = read_jsonl(
    PROCESSED / "products.jsonl"
)

product_rows = []

for item in product_input:

    flat = {}
    flatten("", item, flat)

    product_rows.append({
        "product_name": flat.get("content.productName"),
        "startup_name": flat.get("content.startupName"),
        "pricing_model": flat.get("content.pricingModel"),
        "category": flat.get("content.category"),
        "description": flat.get("content.description"),
        "official_url": flat.get("content.officialUrl"),
        "use_case": flat.get("content.useCase"),
        "features": flat.get("content.features"),
        "source_name": flat.get("source.name"),
        "source_url": flat.get("source.url"),
        "collected_at": flat.get("collectedAt"),
    })


# ============================================================
# RESEARCH PAPERS
# ============================================================

paper_input = read_jsonl(
    PROCESSED / "arxiv_papers_enriched.jsonl"
)

if not paper_input:
    paper_input = read_jsonl(
        RAW / "arxiv_papers.jsonl"
    )

paper_rows = []

for item in paper_input:

    flat = {}
    flatten("", item, flat)

    paper_rows.append({
        "title": (
            flat.get("content.title")
            or flat.get("title")
        ),
        "authors": (
            flat.get("content.authors")
            or flat.get("authors")
        ),
        "paper_url": (
            flat.get("content.paper_url")
            or flat.get("paper_url")
        ),
        "github_url": (
            flat.get("content.github_url")
            or flat.get("github_url")
        ),
        "github_stars": (
            flat.get("content.github_stars")
            if flat.get("content.github_stars") is not None
            else flat.get("github_stars")
        ),
        "published_date": (
            flat.get("content.published_date")
            or flat.get("published_date")
        ),
        "record_type": flat.get("recordType"),
        "source_url": (
            flat.get("source.url")
            or flat.get("source_url")
        ),
    })


# ============================================================
# JOBS AND NEWS
# ============================================================

def generic_rows(filename: str) -> list[dict[str, Any]]:

    result = []

    for item in read_jsonl(PROCESSED / filename):

        flat = {}
        flatten("", item, flat)

        result.append(flat)

    return result


job_rows = generic_rows("jobs.jsonl")
news_rows = generic_rows("news.jsonl")


# ============================================================
# ENTITY MAPPING LOG
# ============================================================

mapping_rows = []

for item in read_jsonl(
    PROCESSED / "entity_mapping_log.jsonl"
):

    flat = {}
    flatten("", item, flat)

    mapping_rows.append(flat)


# ============================================================
# EXCEL EXPORT: SIX REQUIRED TABS
# ============================================================

sheets = [
    ("Startups", startup_rows),
    ("Products", product_rows),
    ("Research Papers", paper_rows),
    ("Jobs", job_rows),
    ("News", news_rows),
    ("Entity Mapping Log", mapping_rows),
]

workbook = Workbook()

workbook.remove(workbook.active)

for title, rows in sheets:

    worksheet = workbook.create_sheet(title)

    headers = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    if not headers:
        headers = ["status"]

    worksheet.append(headers)

    for row in rows:

        worksheet.append([
            json_cell(row.get(header))
            for header in headers
        ])

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    # Header formatting
    for cell in worksheet[1]:

        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            "solid",
            fgColor="243B53"
        )

        cell.alignment = Alignment(
            wrap_text=True,
            vertical="top"
        )

    # Column widths
    for col_idx, header in enumerate(headers, 1):

        sample = [
            str(
                worksheet.cell(row_idx, col_idx).value or ""
            )
            for row_idx in range(
                2,
                min(worksheet.max_row, 30) + 1
            )
        ]

        width = min(
            max(
                [len(str(header))]
                + [min(len(value), 48) for value in sample]
            ) + 2,
            52
        )

        worksheet.column_dimensions[
            get_column_letter(col_idx)
        ].width = max(12, width)

    # Cell formatting
    for row in worksheet.iter_rows(min_row=2):

        for cell in row:

            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True
            )

    safe_name = title.lower().replace(" ", "_")

    write_csv(
        safe_name,
        headers,
        rows
    )


# ============================================================
# SAVE WORKBOOK
# ============================================================

xlsx_path = OUT / "graphone_submission.xlsx"

workbook.save(xlsx_path)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n========== SUBMISSION EXPORT REPORT ==========")

for title, rows in sheets:

    print(f"{title}: {len(rows)} rows")


print(
    "Startup AI_RELEVANT:",
    sum(
        1 for row in startup_rows
        if row["ai_classification"] == "AI_RELEVANT"
    )
)

print(
    "Startup NOT_AI_RELEVANT:",
    sum(
        1 for row in startup_rows
        if row["ai_classification"] == "NOT_AI_RELEVANT"
    )
)

print(
    "Startup NEEDS_REVIEW:",
    sum(
        1 for row in startup_rows
        if row["ai_classification"] == "NEEDS_REVIEW"
    )
)

print(
    "Startup PENDING:",
    sum(
        1 for row in startup_rows
        if row["ai_classification"] == "PENDING"
    )
)

print("\nWorkbook:", xlsx_path)
print("CSV exports:", OUT)

print("==============================================")