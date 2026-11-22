"""
Offline completion path for YC startup classifications.

Uses only existing YC candidate + evidence JSONL files.
Preserves the existing checkpoint and skips every URL already classified.
New records are explicitly tagged as LOCAL_RULES (not an LLM result).
Ambiguous records are NEEDS_REVIEW; absence of AI keywords is not treated
as proof that a company is not AI-related.
"""
import json
import re
from pathlib import Path
from collections import Counter

from src.groq_startup_classifier import (
    CANDIDATES_FILE,
    EVIDENCE_FILE,
    CHECKPOINT_FILE,
    read_jsonl,
    append_jsonl,
    get_name,
    get_url,
    load_evidence,
    extract_evidence,
    get_latest_checkpoint_results,
)
from src.schemas.startup_classification import StartupClassification

AI_PATTERNS = [
    r"\bartificial intelligence\b", r"\bmachine learning\b", r"\bdeep learning\b",
    r"\bneural network(s)?\b", r"\blarge language model(s)?\b", r"\bLLM(s)?\b",
    r"\bgenerative AI\b", r"\bcomputer vision\b", r"\bnatural language processing\b",
    r"\bNLP\b", r"\bspeech recognition\b", r"\bspeech synthesis\b",
    r"\bAI agent(s)?\b", r"\bAI-powered\b", r"\bAI platform\b",
    r"\bAI model(s)?\b", r"\bMLOps\b", r"\bRAG\b", r"\btransformer model(s)?\b",
    r"\bGPT\b", r"\bdiffusion model(s)?\b", r"\breinforcement learning\b",
    r"\bautonomous agent(s)?\b", r"\bintelligent automation\b",
]
CORE_TERMS = [
    "machine learning", "deep learning", "large language model", "generative ai",
    "computer vision", "natural language processing", "speech recognition",
    "ai agent", "ai model", "mlops", "rag", "artificial intelligence",
]

def classify_locally(e):
    text = " ".join([
        e.get("description", ""),
        e.get("one_liner", ""),
        " ".join(e.get("tags", [])),
        " ".join(e.get("industries", [])),
        e.get("subindustry", ""),
    ]).lower()

    hits = [p for p in AI_PATTERNS if re.search(p, text, flags=re.I)]
    matched_core = [term for term in CORE_TERMS if term in text]

    if hits:
        strength = "STRONG" if len(matched_core) >= 2 or any(
            term in text for term in (
                "machine learning", "large language model", "generative ai",
                "computer vision", "deep learning", "artificial intelligence"
            )
        ) else "MODERATE"
        evidence = []
        for field, label in [
            ("one_liner", "Company one-liner"),
            ("description", "Company description"),
        ]:
            value = e.get(field, "").strip()
            if value and any(re.search(p, value, flags=re.I) for p in AI_PATTERNS):
                evidence.append(f"{label}: {value[:350]}")
        for tag in e.get("tags", []):
            if any(re.search(p, tag, flags=re.I) for p in AI_PATTERNS):
                evidence.append(f"YC tag: {tag}")
        if not evidence:
            evidence = ["AI/ML terminology appears in supplied YC evidence."]
        return "AI_RELEVANT", strength, evidence, 0.82 if strength == "STRONG" else 0.70

    # No AI keywords is not sufficient evidence to assert NOT_AI_RELEVANT.
    snippets = []
    if e.get("one_liner"):
        snippets.append("YC one-liner: " + e["one_liner"][:300])
    if e.get("description"):
        snippets.append("YC description: " + e["description"][:300])
    if e.get("tags"):
        snippets.append("YC tags: " + ", ".join(e["tags"][:8]))
    return "NEEDS_REVIEW", "WEAK", snippets or [
        "Insufficient descriptive evidence for a reliable AI relevance decision."
    ], 0.35

def main():
    candidates = read_jsonl(CANDIDATES_FILE)
    evidence_map = load_evidence()
    latest = get_latest_checkpoint_results()
    completed = set(latest)

    pending, seen = [], set()
    for c in candidates:
        name, url = get_name(c), get_url(c)
        if not name or not url:
            continue
        norm = url.rstrip("/")
        if norm in seen:
            continue
        seen.add(norm)
        if norm in completed:
            continue
        ev = extract_evidence(c, evidence_map.get(norm, {}))
        pending.append((name, url, ev))

    print(f"Unique candidates: {len(seen)}")
    print(f"Already classified (preserved): {len(completed)}")
    print(f"To process locally: {len(pending)}")

    counts = Counter()
    for i, (name, url, ev) in enumerate(pending, 1):
        classification, strength, evidence, confidence = classify_locally(ev)
        result = StartupClassification(
            entity_name=name,
            source_url=url,
            classification=classification,
            evidence_strength=strength,
            reasoning=(
                "Offline rule-based classification using only the supplied "
                "Y Combinator one-liner, description, tags, and industry fields. "
                + (
                    "Explicit AI/ML evidence was found."
                    if classification == "AI_RELEVANT"
                    else "No decisive AI/ML evidence was found; manual review is required."
                )
            ),
            evidence=evidence,
            confidence=confidence,
        ).model_dump(mode="json")

        append_jsonl(CHECKPOINT_FILE, {
            "status": "completed",
            "prompt_version": "local-rules-v1",
            "model_name": "LOCAL_RULES_NOT_LLM",
            "source_url": url,
            "result": result,
        })
        counts[classification] += 1
        if i % 50 == 0 or i == len(pending):
            print(f"Processed {i}/{len(pending)}")

    final = get_latest_checkpoint_results()
    final_counts = Counter(x["result"]["classification"] for x in final.values())
    print("\n=== OFFLINE CLASSIFICATION COMPLETE ===")
    print(f"New local records: {sum(counts.values())}")
    print(f"New local classification counts: {dict(counts)}")
    print(f"Total unique classified URLs: {len(final)}")
    print(f"Still unclassified: {max(0, len(seen) - len(final))}")
    print(f"Latest classification counts: {dict(final_counts)}")
    print("Local results are labelled LOCAL_RULES_NOT_LLM.")

if __name__ == "__main__":
    main()
