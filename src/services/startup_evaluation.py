
import json
from pathlib import Path

from src.services.startup_batch_classifier import read_jsonl
from src.services.gemini_classifier import classify_startup


INPUT_FILE = Path("data/raw/yc_startup_evidence.jsonl")

OUTPUT_FILE = Path(
    "data/processed/startup_evaluation_v2.jsonl"
)

SAMPLE_SIZE = 10


def main():
    companies = read_jsonl(INPUT_FILE)

    if not companies:
        raise RuntimeError("No startup evidence found.")

    # Use a separate output file for the revised prompt.
    # Never overwrite the original checkpoint.
    if OUTPUT_FILE.exists():
        raise RuntimeError(
            f"{OUTPUT_FILE} already exists. "
            "Preserve it or choose another evaluation filename."
        )

    sample = companies[:SAMPLE_SIZE]

    print(f"Evaluation sample: {len(sample)} companies")

    for index, company in enumerate(sample, start=1):
        name = company.get("entityName", "Unknown")

        print(f"\n[{index}/{len(sample)}] {name}")

        try:
            result = classify_startup(company)

            record = {
                "prompt_version": "2.0",
                "model_name": "gemini-3.5-flash-lite",
                "source_url": company["sourceUrl"],
                "result": result.model_dump(mode="json"),
            }

            OUTPUT_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with OUTPUT_FILE.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

            print(
                f"{result.classification.value} "
                f"| Confidence: {result.confidence}"
            )

        except Exception as exc:
            print(f"FAILED: {name} | {exc}")

    print("\nEvaluation completed.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
