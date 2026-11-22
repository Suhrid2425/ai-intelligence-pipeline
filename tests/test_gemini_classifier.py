from src.services.gemini_classifier import classify_startup


def test_marft_classification():
    company = {
        "entityName": "Marft",
        "sourceUrl": "https://www.ycombinator.com/companies/marft",
        "oneLiner": (
            "Marft creates embeddable machine learning models "
            "for application developers."
        ),
        "description": (
            "Marft creates embeddable machine learning models "
            "for application developers. Users submit data by "
            "web/API, and receive optimized models implemented "
            "in a language of their choice."
        ),
        "tags": ["Developer Tools", "Machine Learning"],
        "industries": ["B2B", "Engineering, Product and Design"],
        "subindustry": "B2B -> Engineering, Product and Design",
    }

    result = classify_startup(company)

    print("\nClassification:", result.model_dump_json(indent=2))

    assert result.entity_name == "Marft"
    assert result.classification.value in {
        "AI_RELEVANT",
        "NOT_AI_RELEVANT",
        "NEEDS_REVIEW",
    }
    assert 0 <= result.confidence <= 1