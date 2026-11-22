
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse
from openpyxl import load_workbook

FILE = Path("submission/graphone_submission.xlsx")
wb = load_workbook(FILE, read_only=True, data_only=True)

def valid_url(value):
    if not value:
        return False
    try:
        parsed = urlparse(str(value).strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False

def inspect_sheet(name, required=(), unique_key=None):
    ws = wb[name]
    rows = list(ws.iter_rows(values_only=True))
    headers = list(rows[0])
    records = [dict(zip(headers, row)) for row in rows[1:]]
    print(f"\n--- {name} ---")
    print("Records:", len(records))

    for field in required:
        if field not in headers:
            print(f"[MISSING COLUMN] {field}")
            continue
        missing = sum(
            1 for r in records
            if r.get(field) is None or str(r.get(field)).strip() == ""
        )
        print(f"{field}: missing={missing}")

    url_fields = [h for h in headers if "url" in str(h).lower()]
    for field in url_fields:
        values = [r.get(field) for r in records]
        present = [v for v in values if v]
        invalid = sum(1 for v in present if not valid_url(v))
        print(
            f"{field}: present={len(present)}, "
            f"missing={len(records)-len(present)}, invalid={invalid}"
        )

    if unique_key and unique_key in headers:
        values = [
            str(r[unique_key]).strip().lower()
            for r in records if r.get(unique_key)
        ]
        duplicates = len(values) - len(set(values))
        print(f"Duplicate {unique_key}: {duplicates}")

    return records, headers

startups, startup_headers = inspect_sheet(
    "Startups",
    required=("startup_name", "source_url", "ai_classification"),
    unique_key="source_url"
)

products, _ = inspect_sheet(
    "Products",
    required=("product_name", "source_url"),
    unique_key="source_url"
)

papers, _ = inspect_sheet(
    "Research Papers",
    required=("title", "paper_url"),
    unique_key="paper_url"
)

jobs, _ = inspect_sheet("Jobs")
news, _ = inspect_sheet("News")

mapping, mapping_headers = inspect_sheet("Entity Mapping Log")

print("\n--- STARTUP CLASSIFICATION PROVENANCE ---")
print("Classification counts:", dict(Counter(
    str(r.get("ai_classification"))
    for r in startups
)))

print("Model counts:", dict(Counter(
    str(r.get("model_name"))
    for r in startups
)))

print("Prompt versions:", dict(Counter(
    str(r.get("prompt_version"))
    for r in startups
)))

bad_confidence = 0
for r in startups:
    value = r.get("confidence")
    if value not in (None, ""):
        try:
            if not 0 <= float(value) <= 1:
                bad_confidence += 1
        except (ValueError, TypeError):
            bad_confidence += 1

print("Invalid confidence values:", bad_confidence)

print("\n--- AUDIT FINISHED ---")
wb.close()
