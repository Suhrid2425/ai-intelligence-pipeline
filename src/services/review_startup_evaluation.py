
import json
from pathlib import Path

INPUT_FILE = Path("data/raw/yc_startup_evidence.jsonl")
RESULT_FILE = Path("data/processed/startup_evaluation_v2.jsonl")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    companies = read_jsonl(INPUT_FILE)
    results = read_jsonl(RESULT_FILE)

    evidence_by_url = {
        company["sourceUrl"]: company
        for company in companies
    }

    for index, record in enumerate(results, start=1):
        result = record["result"]
        evidence = evidence_by_url.get(record["source_url"], {})

        print("\n" + "=" * 75)
        print(f"{index}. {result['entity_name']}")
        print("=" * 75)

        print("CLASSIFICATION:", result["classification"])
        print("CONFIDENCE:", result["confidence"])

        print("\nONE-LINER:")
        print(evidence.get("oneLiner"))

        print("\nDESCRIPTION:")
        print(evidence.get("description"))

        print("\nTAGS:")
        print(evidence.get("tags"))

        print("\nMODEL REASONING:")
        print(result["reasoning"])

        print("\nMODEL EVIDENCE:")
        for item in result["evidence"]:
            print("-", item)


if __name__ == "__main__":
    main()
