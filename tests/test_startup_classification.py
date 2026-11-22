
import pytest
from pydantic import ValidationError

from src.schemas.startup_classification import (
    AIRelevance,
    EvidenceStrength,
    StartupClassification,
)


def test_valid_classification():
    result = StartupClassification(
        entity_name="Marft",
        source_url="https://www.ycombinator.com/companies/marft",
        classification=AIRelevance.AI_RELEVANT,
        evidence_strength=EvidenceStrength.STRONG,
        reasoning="The company creates machine learning models.",
        evidence=["Machine Learning"],
        confidence=0.98,
    )

    assert result.entity_name == "Marft"
    assert result.classification == AIRelevance.AI_RELEVANT
    assert result.evidence_strength == EvidenceStrength.STRONG
    assert result.confidence == 0.98


def test_all_classification_categories():
    for category in AIRelevance:
        result = StartupClassification(
            entity_name="Example",
            source_url="https://example.com",
            classification=category,
            evidence_strength=EvidenceStrength.MODERATE,
            reasoning="Test reasoning",
            evidence=["Test evidence"],
            confidence=0.5,
        )

        assert result.classification == category


def test_all_evidence_strength_categories():
    for strength in EvidenceStrength:
        result = StartupClassification(
            entity_name="Example",
            source_url="https://example.com",
            classification=AIRelevance.AI_RELEVANT,
            evidence_strength=strength,
            reasoning="Test reasoning",
            confidence=0.5,
        )

        assert result.evidence_strength == strength


def test_confidence_cannot_exceed_one():
    with pytest.raises(ValidationError):
        StartupClassification(
            entity_name="Example",
            source_url="https://example.com",
            classification=AIRelevance.AI_RELEVANT,
            evidence_strength=EvidenceStrength.STRONG,
            reasoning="Test reasoning",
            confidence=1.5,
        )


def test_confidence_cannot_be_negative():
    with pytest.raises(ValidationError):
        StartupClassification(
            entity_name="Example",
            source_url="https://example.com",
            classification=AIRelevance.AI_RELEVANT,
            evidence_strength=EvidenceStrength.STRONG,
            reasoning="Test reasoning",
            confidence=-0.1,
        )


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        StartupClassification(
            entity_name="Example",
            source_url="https://example.com",
            classification=AIRelevance.AI_RELEVANT,
            evidence_strength=EvidenceStrength.STRONG,
            reasoning="Test reasoning",
            confidence=0.5,
            unknown_field="not allowed",
        )


def test_evidence_strength_is_required():
    with pytest.raises(ValidationError):
        StartupClassification(
            entity_name="Example",
            source_url="https://example.com",
            classification=AIRelevance.AI_RELEVANT,
            reasoning="Test reasoning",
            confidence=0.5,
        )
