"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    app_name: str = "Enterprise Knowledge Agent"
    app_environment: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EKA_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for the application process."""

    return Settings()
