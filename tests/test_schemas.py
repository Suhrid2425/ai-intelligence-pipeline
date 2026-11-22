
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.schemas.entities import (
    StartupEntity,
    ProductEntity,
    ResearchPaperEntity,
    JobEntity,
)


def test_startup_entity():
    startup = StartupEntity(
        source={
            "name": "Example Directory",
            "url": "https://example.com/startup",
        },
        content={
            "entityName": "OpenAI",
            "data": {"employeeCount": 1000},
        },
        collectedAt=datetime.now(timezone.utc),
    )

    assert startup.recordType == "STARTUP"
    assert startup.content.entityName == "OpenAI"


def test_product_entity():
    product = ProductEntity(
        source={
            "name": "Example Directory",
            "url": "https://example.com/product",
        },
        content={
            "startupName": "OpenAI",
            "pricingModel": "FREEMIUM",
        },
        collectedAt=datetime.now(timezone.utc),
    )

    assert product.content.pricingModel.value == "FREEMIUM"


def test_research_paper_entity():
    paper = ResearchPaperEntity(
        content={
            "title": "Attention Is All You Need",
            "authors": ["Author One"],
            "paper_url": "https://arxiv.org/abs/1706.03762",
            "github_url": "https://github.com/example/repository",
            "github_stars": 100,
            "published_date": datetime.now(timezone.utc),
        }
    )

    assert paper.recordType == "RESEARCH_PAPER"
    assert paper.content.github_stars == 100


def test_job_entity():
    job = JobEntity(
        content={
            "company": "Example AI",
            "date": datetime.now(timezone.utc),
            "is_remote": True,
            "role_family": "Engineering",
        }
    )

    assert job.recordType == "JOB"
    assert job.content.is_remote is True


def test_invalid_pricing_model():
    with pytest.raises(ValidationError):
        ProductEntity(
            source={
                "name": "Example",
                "url": "https://example.com",
            },
            content={
                "startupName": "OpenAI",
                "pricingModel": "UNKNOWN",
            },
            collectedAt=datetime.now(timezone.utc),
        )
def test_startup_entity_valid():
    from src.schemas.entities import StartupEntity

    startup = StartupEntity(
        source={
            "name": "Example Startup Directory",
            "url": "https://example.com/startups",
        },
        content={
            "entityName": "OpenAI",
            "data": {
                "employeeCount": 1000,
            },
        },
        collectedAt="2026-11-21T20:00:00Z",
    )

    assert startup.recordType == "STARTUP"
    assert startup.content.entityName == "OpenAI"
    assert startup.content.data.employeeCount == 1000


def test_startup_employee_count_can_be_missing():
    from src.schemas.entities import StartupEntity

    startup = StartupEntity(
        source={
            "name": "Example Directory",
            "url": "https://example.com",
        },
        content={
            "entityName": "Example AI",
            "data": {},
        },
        collectedAt="2026-11-21T20:00:00Z",
    )

    assert startup.content.data.employeeCount is None


def test_startup_employee_count_can_be_zero():
    from src.schemas.entities import StartupEntity

    startup = StartupEntity(
        source={
            "name": "Example Directory",
            "url": "https://example.com",
        },
        content={
            "entityName": "Early Stage AI",
            "data": {
                "employeeCount": 0,
            },
        },
        collectedAt="2026-11-21T20:00:00Z",
    )

    assert startup.content.data.employeeCount == 0


def test_startup_employee_count_rejects_negative():
    import pytest
    from pydantic import ValidationError
    from src.schemas.entities import StartupEntity

    with pytest.raises(ValidationError):
        StartupEntity(
            source={
                "name": "Example Directory",
                "url": "https://example.com",
            },
            content={
                "entityName": "Invalid Startup",
                "data": {
                    "employeeCount": -5,
                },
            },
            collectedAt="2026-11-21T20:00:00Z",
        )


def test_startup_record_type_is_fixed():
    import pytest
    from pydantic import ValidationError
    from src.schemas.entities import StartupEntity

    with pytest.raises(ValidationError):
        StartupEntity(
            recordType="PRODUCT",
            source={
                "name": "Example Directory",
                "url": "https://example.com",
            },
            content={
                "entityName": "Example AI",
                "data": {},
            },
            collectedAt="2026-11-21T20:00:00Z",
        )