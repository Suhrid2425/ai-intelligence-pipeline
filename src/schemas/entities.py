from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class BaseEntity(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    schemaVersion: str = "1.0"


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: HttpUrl


class StartupContentData(BaseModel):
    employeeCount: int | None = Field(default=None, ge=0)


class StartupContent(BaseModel):
    entityName: str = Field(min_length=1)
    data: StartupContentData = Field(default_factory=StartupContentData)


class StartupEntity(BaseEntity):
    recordType: Literal["STARTUP"] = "STARTUP"
    source: SourceInfo
    content: StartupContent
    collectedAt: datetime


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"


class ProductContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    productName: str | None = None
    startupName: str | None = None
    pricingModel: PricingModel | None = None
    category: str | None = None
    description: str | None = None
    officialUrl: HttpUrl | None = None
    useCase: str | None = None
    features: list[str] = Field(default_factory=list)
    added: str | None = None


class ProductEntity(BaseEntity):
    recordType: Literal["PRODUCT"] = "PRODUCT"
    source: SourceInfo
    content: ProductContent
    collectedAt: datetime


class ResearchPaperContent(BaseModel):
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    paper_url: HttpUrl
    github_url: HttpUrl | None = None
    github_stars: int | None = Field(default=None, ge=0)
    published_date: datetime


class ResearchPaperEntity(BaseEntity):
    recordType: Literal["RESEARCH_PAPER"] = "RESEARCH_PAPER"
    content: ResearchPaperContent


class JobContent(BaseModel):
    company: str = Field(min_length=1)
    date: datetime
    is_remote: bool
    role_family: str = Field(min_length=1)


class JobEntity(BaseEntity):
    recordType: Literal["JOB"] = "JOB"
    content: JobContent