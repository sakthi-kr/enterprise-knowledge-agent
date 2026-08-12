"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from enterprise_knowledge_agent.agent import EnterpriseKnowledgeAgent
from enterprise_knowledge_agent.api_runtime import (
    ApiError,
    api_error_response,
    bind_request_id,
    configure_logging,
    current_request_id,
    normalize_request_id,
    reset_request_id,
)
from enterprise_knowledge_agent.config import Settings, get_settings
from enterprise_knowledge_agent.grounded_answer import GroundedAnswer, GroundedAnswerService
from enterprise_knowledge_agent.language_model_errors import LanguageModelAPIError
from enterprise_knowledge_agent.qdrant_store import QdrantStoreError
from enterprise_knowledge_agent.readiness import ReadinessReport, check_readiness
from enterprise_knowledge_agent.runtime import (
    ServiceConfigurationError,
    close_runtime_services,
    get_agent_service,
    get_answer_service,
)

LOGGER = logging.getLogger(__name__)
REQUEST_ID_HEADER = "X-Request-ID"


class HealthResponse(BaseModel):
    """Response returned by the liveness endpoint."""

    status: str
    service: str
    environment: str


class DependencyReadinessResponse(BaseModel):
    """Public readiness state for one required dependency."""

    status: str
    detail: str


class ReadinessResponse(BaseModel):
    """Response returned by the dependency readiness endpoint."""

    status: str
    service: str
    environment: str
    dependencies: dict[str, DependencyReadinessResponse]


class AskRequest(BaseModel):
    """Grounded enterprise question."""

    question: str = Field(min_length=1, max_length=2000)


class CitationResponse(BaseModel):
    """Evidence source cited by a grounded answer."""

    citation_id: str
    rank: int
    score: float
    chunk_id: str
    record_id: str
    doc_id: str
    source_type: str
    title: str
    source_file: str
    text: str
    retrieval_source: str


class TokenUsageResponse(BaseModel):
    """Provider token usage for a generated answer."""

    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    """Grounded answer with validated enterprise citations."""

    status: str
    answer: str
    citations: list[CitationResponse]
    model_name: str
    usage: TokenUsageResponse
    retrieved_chunk_count: int
    context_source_count: int
    retrieval_strategy: str
    graph_context_source_count: int
    graph_candidate_count: int


class AgentPlanResponse(BaseModel):
    """LLM planner decision returned with an agent response."""

    strategy: str
    reason: str
    model_name: str
    usage: TokenUsageResponse


class ToolExecutionResponse(BaseModel):
    """One retrieval tool execution performed by the agent."""

    tool_name: str
    status: str
    result_count: int
    detail: str


class AgentAskResponse(BaseModel):
    """Agent answer plus planner and tool-execution metadata."""

    answer: AskResponse
    plan: AgentPlanResponse
    tool_trace: list[ToolExecutionResponse]
    planner_fallback: bool
    tool_call_count: int


def provide_answer_service() -> GroundedAnswerService:
    """Resolve the cached RAG service and surface configuration errors cleanly."""

    try:
        return get_answer_service()
    except ServiceConfigurationError as exc:
        raise ApiError(
            status_code=503,
            code="service_not_configured",
            message=str(exc),
        ) from exc


def provide_agent_service() -> EnterpriseKnowledgeAgent:
    """Resolve the cached agent service and surface configuration errors cleanly."""

    try:
        return get_agent_service()
    except ServiceConfigurationError as exc:
        raise ApiError(
            status_code=503,
            code="service_not_configured",
            message=str(exc),
        ) from exc


def _ask_payload(result: GroundedAnswer) -> AskResponse:
    payload = asdict(result)
    payload["status"] = result.status.value
    return AskResponse.model_validate(payload)


def _readiness_payload(settings: Settings, report: ReadinessReport) -> ReadinessResponse:
    return ReadinessResponse(
        status="ready" if report.ready else "not_ready",
        service=settings.app_name,
        environment=settings.app_environment,
        dependencies={
            name: DependencyReadinessResponse(status=item.status, detail=item.detail)
            for name, item in report.dependencies.items()
        },
    )


def create_app(
    *,
    settings: Settings | None = None,
    readiness_checker: Callable[[Settings], ReadinessReport] = check_readiness,
) -> FastAPI:
    """Create and configure the API application."""

    resolved_settings = settings or get_settings()
    if resolved_settings.app_request_timeout_seconds <= 0:
        raise ValueError("app_request_timeout_seconds must be greater than zero")
    if resolved_settings.app_readiness_timeout_seconds <= 0:
        raise ValueError("app_readiness_timeout_seconds must be greater than zero")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_logging(resolved_settings.log_level)
        LOGGER.info(
            "application started",
            extra={
                "structured": {
                    "service": resolved_settings.app_name,
                    "environment": resolved_settings.app_environment,
                }
            },
        )
        try:
            yield
        finally:
            close_runtime_services()
            LOGGER.info(
                "application stopped",
                extra={"structured": {"service": resolved_settings.app_name}},
            )

    application = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        description="Enterprise knowledge agent API.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        token = bind_request_id(request_id)
        started = time.perf_counter()
        response: Response
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=resolved_settings.app_request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            response = api_error_response(
                status_code=504,
                code="request_timeout",
                message="The request exceeded the configured processing timeout.",
                request_id=request_id,
            )
        except Exception as exc:
            LOGGER.error(
                "unhandled request failure",
                extra={"structured": {"error_type": type(exc).__name__}},
                exc_info=True,
            )
            response = api_error_response(
                status_code=500,
                code="internal_error",
                message="The service could not complete the request.",
                request_id=request_id,
            )
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            reset_request_id(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        LOGGER.info(
            "request completed",
            extra={
                "request_id": request_id,
                "structured": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            },
        )
        return response

    @application.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return api_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            request_id=getattr(request.state, "request_id", current_request_id()),
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = str(exc.detail) if isinstance(exc.detail, str) else "HTTP request failed."
        code = "not_found" if exc.status_code == 404 else "http_error"
        return api_error_response(
            status_code=exc.status_code,
            code=code,
            message=message,
            request_id=getattr(request.state, "request_id", current_request_id()),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        _: RequestValidationError,
    ) -> JSONResponse:
        return api_error_response(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            request_id=getattr(request.state, "request_id", current_request_id()),
        )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=resolved_settings.app_name,
            environment=resolved_settings.app_environment,
        )

    @application.get("/ready", response_model=ReadinessResponse, tags=["system"])
    def ready(response: Response) -> ReadinessResponse:
        report = readiness_checker(resolved_settings)
        if not report.ready:
            response.status_code = 503
        return _readiness_payload(resolved_settings, report)

    @application.post("/ask", response_model=AskResponse, tags=["knowledge"])
    def ask(
        request: AskRequest,
        service: Annotated[GroundedAnswerService, Depends(provide_answer_service)],
    ) -> AskResponse:
        try:
            result = service.answer(request.question)
        except QdrantStoreError as exc:
            raise ApiError(
                status_code=503,
                code="retrieval_unavailable",
                message="The retrieval service is unavailable.",
            ) from exc
        except LanguageModelAPIError as exc:
            raise ApiError(
                status_code=502,
                code="language_model_unavailable",
                message="The language-model provider could not complete the request.",
            ) from exc
        except RuntimeError as exc:
            raise ApiError(
                status_code=502,
                code="generation_failed",
                message="Grounded answer generation failed.",
            ) from exc
        return _ask_payload(result)

    @application.post("/agent/ask", response_model=AgentAskResponse, tags=["agent"])
    def agent_ask(
        request: AskRequest,
        service: Annotated[EnterpriseKnowledgeAgent, Depends(provide_agent_service)],
    ) -> AgentAskResponse:
        try:
            result = service.run(request.question)
        except QdrantStoreError as exc:
            raise ApiError(
                status_code=503,
                code="retrieval_unavailable",
                message="The retrieval service is unavailable.",
            ) from exc
        except LanguageModelAPIError as exc:
            raise ApiError(
                status_code=502,
                code="language_model_unavailable",
                message="The language-model provider could not complete the request.",
            ) from exc
        except RuntimeError as exc:
            raise ApiError(
                status_code=502,
                code="agent_execution_failed",
                message="The agent could not complete the request.",
            ) from exc

        return AgentAskResponse(
            answer=_ask_payload(result.answer),
            plan=AgentPlanResponse(
                strategy=result.plan.strategy.value,
                reason=result.plan.reason,
                model_name=result.plan.model_name,
                usage=TokenUsageResponse.model_validate(asdict(result.plan.usage)),
            ),
            tool_trace=[
                ToolExecutionResponse.model_validate(asdict(item)) for item in result.tool_trace
            ],
            planner_fallback=result.planner_fallback,
            tool_call_count=result.tool_call_count,
        )

    return application


app = create_app()
