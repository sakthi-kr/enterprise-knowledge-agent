"""FastAPI application entry point."""

from fastapi import FastAPI
from pydantic import BaseModel

from enterprise_knowledge_agent.config import get_settings


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str
    service: str
    environment: str


def create_app() -> FastAPI:
    """Create and configure the API application."""

    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Enterprise knowledge agent API.",
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            environment=settings.app_environment,
        )

    return application


app = create_app()
