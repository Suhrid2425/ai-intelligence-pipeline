
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AIRelevance(str, Enum):
    AI_RELEVANT = "AI_RELEVANT"
    NOT_AI_RELEVANT = "NOT_AI_RELEVANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class EvidenceStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class StartupClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)

    classification: AIRelevance
    evidence_strength: EvidenceStrength

    reasoning: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
