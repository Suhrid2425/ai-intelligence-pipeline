
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration."""

    # Application
    app_name: str = "AI Intelligence Pipeline"
    app_env: str = "development"
    debug: bool = True

    # LLM API keys
    gemini_api_key: str = ""
    groq_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    groq_model: str = "openai/gpt-oss-120b"
    deepseek_model: str = "deepseek-flash"
    github_token: str = ""
    # Database
    database_url: str = ""

    # Pipeline configuration
    max_concurrency: int = 10
    request_timeout: int = 30
    max_retries: int = 3

    # Data collection
    arxiv_batch_size: int = 100
    target_papers: int = 1000
    target_startups: int = 1000
    target_products: int = 1000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()


settings = get_settings()
