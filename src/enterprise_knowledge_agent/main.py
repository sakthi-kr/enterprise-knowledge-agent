"""FastAPI application entry point."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from enterprise_knowledge_agent.agent import EnterpriseKnowledgeAgent
from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.gemini_client import GeminiAPIError
from enterprise_knowledge_agent.grounded_answer import GroundedAnswer, GroundedAnswerService
from enterprise_knowledge_agent.qdrant_store import QdrantStoreError
from enterprise_knowledge_agent.runtime import (
    ServiceConfigurationError,
    get_agent_service,
    get_answer_service,
)


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str
    service: str
    environment: str


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
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def provide_agent_service() -> EnterpriseKnowledgeAgent:
    """Resolve the cached agent service and surface configuration errors cleanly."""

    try:
        return get_agent_service()
    except ServiceConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _ask_payload(result: GroundedAnswer) -> AskResponse:
    payload = asdict(result)
    payload["status"] = result.status.value
    return AskResponse.model_validate(payload)


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

    @application.post("/ask", response_model=AskResponse, tags=["knowledge"])
    def ask(
        request: AskRequest,
        service: Annotated[GroundedAnswerService, Depends(provide_answer_service)],
    ) -> AskResponse:
        try:
            result = service.answer(request.question)
        except QdrantStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GeminiAPIError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return _ask_payload(result)

    @application.post("/agent/ask", response_model=AgentAskResponse, tags=["agent"])
    def agent_ask(
        request: AskRequest,
        service: Annotated[EnterpriseKnowledgeAgent, Depends(provide_agent_service)],
    ) -> AgentAskResponse:
        try:
            result = service.run(request.question)
        except QdrantStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GeminiAPIError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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
