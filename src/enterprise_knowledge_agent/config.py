"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    app_name: str = "Enterprise Knowledge Agent"
    app_environment: str = "development"
    log_level: str = "INFO"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "enterprise_knowledge_chunks"
    qdrant_timeout_seconds: float = 120.0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EKA_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for the application process."""

    return Settings()
