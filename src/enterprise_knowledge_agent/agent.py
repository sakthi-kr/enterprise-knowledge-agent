"""LangGraph orchestration for bounded enterprise retrieval tools."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from enterprise_knowledge_agent.agent_types import (
    AgentPlan,
    AgentResult,
    AgentStrategy,
    ToolExecution,
)
from enterprise_knowledge_agent.graph_retrieval import GraphRAGRetriever, GraphRetrievalTrace
from enterprise_knowledge_agent.grounded_answer import (
    GraphContextBuilder,
    GroundedAnswer,
    GroundedAnswerService,
    TokenUsage,
)
from enterprise_knowledge_agent.language_model_errors import LanguageModelAPIError
from enterprise_knowledge_agent.observability import NullTracer, Tracer, TraceSpan
from enterprise_knowledge_agent.vector_search import RetrievalHit, VectorRetriever


class AgentPlanner(Protocol):
    """Planner interface required by the agent workflow."""

    def plan(self, *, question: str) -> AgentPlan:
        """Select a retrieval strategy for one question."""


class AgentState(TypedDict, total=False):
    """Explicit state carried between LangGraph nodes."""

    question: str
    plan: AgentPlan
    dense_hits: list[RetrievalHit]
    graph_hits: list[RetrievalHit]
    graph_trace: GraphRetrievalTrace
    graph_candidate_count: int
    tool_trace: list[ToolExecution]
    tool_call_count: int
    planner_fallback: bool
    answer: GroundedAnswer


def _token_usage_dict(usage: TokenUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _set_llm_span_attributes(
    span: TraceSpan,
    *,
    model_name: str,
    provider_name: str,
    usage: TokenUsage,
) -> None:
    span.set_attribute("mlflow.llm.model", model_name)
    span.set_attribute("mlflow.llm.provider", provider_name)
    span.set_attribute("mlflow.chat.tokenUsage", _token_usage_dict(usage))
    span.set_attribute("enterprise.thinking_tokens", usage.thinking_tokens)


def _trace_text_metadata(text: str) -> dict[str, int]:
    """Return bounded metadata without copying potentially sensitive text into traces."""

    return {"character_count": len(text)}


def _retriever_documents(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    """Return a content-redacted MLflow retriever shape for dense retrieval spans."""

    documents: list[dict[str, Any]] = []
    for hit in hits:
        documents.append(
            {
                "page_content": "",
                "metadata": {
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "record_id": hit.record_id,
                    "source_type": hit.source_type,
                    "score": hit.score,
                },
                "id": hit.chunk_id,
            }
        )
    return documents


class EnterpriseKnowledgeAgent:
    """Plan and execute a bounded dense/graph retrieval workflow."""

    def __init__(
        self,
        *,
        planner: AgentPlanner,
        dense_retriever: VectorRetriever,
        graph_retriever: GraphRAGRetriever,
        answer_service: GroundedAnswerService,
        context_builder: GraphContextBuilder,
        retrieval_candidates: int = 12,
        graph_document_candidates: int = 10,
        graph_fetch_candidates: int = 4,
        min_graph_matched_entities: int = 2,
        max_tool_calls: int = 2,
        tracer: Tracer | None = None,
    ) -> None:
        for name, value in (
            ("retrieval_candidates", retrieval_candidates),
            ("graph_document_candidates", graph_document_candidates),
            ("graph_fetch_candidates", graph_fetch_candidates),
            ("min_graph_matched_entities", min_graph_matched_entities),
            ("max_tool_calls", max_tool_calls),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        self._planner = planner
        self._llm_provider_name = str(getattr(planner, "provider_name", "unknown"))
        self._dense_retriever = dense_retriever
        self._graph_retriever = graph_retriever
        self._answer_service = answer_service
        self._context_builder = context_builder
        self._retrieval_candidates = retrieval_candidates
        self._graph_document_candidates = graph_document_candidates
        self._graph_fetch_candidates = graph_fetch_candidates
        self._min_graph_matched_entities = min_graph_matched_entities
        self._max_tool_calls = max_tool_calls
        self._tracer = tracer or NullTracer()
        self._workflow = self._compile_workflow()

    def run(self, question: str) -> AgentResult:
        """Execute one agent workflow and return its answer plus execution metadata."""

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        with self._tracer.span(
            "enterprise_agent",
            span_type="AGENT",
            inputs={"question": _trace_text_metadata(question)},
        ) as span:
            final_state = self._workflow.invoke(
                {
                    "question": question,
                    "tool_trace": [],
                    "tool_call_count": 0,
                    "planner_fallback": False,
                }
            )
            answer = final_state.get("answer")
            plan = final_state.get("plan")
            if not isinstance(answer, GroundedAnswer) or not isinstance(plan, AgentPlan):
                raise RuntimeError("Agent workflow completed without a valid answer or plan")

            result = AgentResult(
                answer=answer,
                plan=plan,
                tool_trace=tuple(final_state.get("tool_trace", [])),
                planner_fallback=bool(final_state.get("planner_fallback", False)),
                tool_call_count=int(final_state.get("tool_call_count", 0)),
            )
            span.set_outputs(
                {
                    "status": result.answer.status.value,
                    "retrieval_strategy": result.answer.retrieval_strategy,
                    "plan_strategy": result.plan.strategy.value,
                    "planner_fallback": result.planner_fallback,
                    "tool_call_count": result.tool_call_count,
                    "citation_count": len(result.answer.citations),
                }
            )
            span.set_attribute("enterprise.plan_strategy", result.plan.strategy.value)
            span.set_attribute("enterprise.tool_call_count", result.tool_call_count)
            span.set_attribute("enterprise.answer_status", result.answer.status.value)
            return result

    def _compile_workflow(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "Agent orchestration requires LangGraph. Reinstall the project dependencies."
            ) from exc

        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("dense_search", self._dense_search_node)
        builder.add_node("graph_search", self._graph_search_node)
        builder.add_node("synthesize", self._synthesize_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "dense_search")
        builder.add_conditional_edges(
            "dense_search",
            self._route_after_dense,
            {
                "graph_search": "graph_search",
                "synthesize": "synthesize",
            },
        )
        builder.add_edge("graph_search", "synthesize")
        builder.add_edge("synthesize", END)
        return builder.compile()

    def _plan_node(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        with self._tracer.span(
            "plan",
            span_type="LLM",
            inputs={"question": _trace_text_metadata(question)},
            attributes={"enterprise.operation": "agent_planning"},
        ) as span:
            try:
                plan = self._planner.plan(question=question)
                fallback = False
            except LanguageModelAPIError:
                plan = AgentPlan(
                    strategy=AgentStrategy.DENSE_ONLY,
                    reason="Planner unavailable; dense retrieval selected as the safe fallback.",
                    model_name="fallback",
                    usage=TokenUsage(),
                )
                fallback = True

            span.set_outputs(
                {
                    "strategy": plan.strategy.value,
                    "fallback": fallback,
                }
            )
            if not fallback:
                _set_llm_span_attributes(
                    span,
                    model_name=plan.model_name,
                    provider_name=self._llm_provider_name,
                    usage=plan.usage,
                )
            return {"plan": plan, "planner_fallback": fallback}

    def _dense_search_node(self, state: AgentState) -> dict[str, Any]:
        with self._tracer.span(
            "dense_search",
            span_type="RETRIEVER",
            inputs={
                "query": _trace_text_metadata(state["question"]),
                "limit": self._retrieval_candidates,
            },
            attributes={"enterprise.store": "qdrant"},
        ) as span:
            hits = self._dense_retriever.search(
                state["question"],
                limit=self._retrieval_candidates,
            )
            span.set_outputs(_retriever_documents(hits))
            trace = list(state.get("tool_trace", []))
            trace.append(
                ToolExecution(
                    tool_name="dense_search",
                    status="ok",
                    result_count=len(hits),
                    detail="Qdrant semantic chunk retrieval",
                )
            )
            return {
                "dense_hits": hits,
                "tool_trace": trace,
                "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
            }

    def _route_after_dense(self, state: AgentState) -> str:
        plan = state["plan"]
        tool_call_count = int(state.get("tool_call_count", 0))
        if (
            plan.strategy is AgentStrategy.DENSE_PLUS_GRAPH
            and tool_call_count < self._max_tool_calls
        ):
            return "graph_search"
        return "synthesize"

    def _graph_search_node(self, state: AgentState) -> dict[str, Any]:
        with self._tracer.span(
            "graph_expand",
            span_type="TOOL",
            inputs={
                "question": _trace_text_metadata(state["question"]),
                "candidate_limit": self._graph_document_candidates,
            },
            attributes={"enterprise.store": "neo4j"},
        ) as span:
            trace = list(state.get("tool_trace", []))
            dense_hits = list(state.get("dense_hits", []))
            try:
                graph_docs, graph_trace = self._graph_retriever.search_documents_with_trace(
                    state["question"],
                    limit=self._graph_document_candidates,
                )
                dense_record_ids = {hit.record_id for hit in dense_hits}
                graph_record_ids: list[str] = []
                for hit in graph_docs:
                    if "graph" not in hit.retrieval_sources:
                        continue
                    if hit.record_id in dense_record_ids:
                        continue
                    if hit.matched_entity_count < self._min_graph_matched_entities:
                        continue
                    graph_record_ids.append(hit.record_id)
                    if len(graph_record_ids) >= self._graph_fetch_candidates:
                        break

                graph_hits = self._dense_retriever.search_records(
                    state["question"],
                    record_ids=graph_record_ids,
                    limit=len(graph_record_ids) if graph_record_ids else None,
                )
                detail = (
                    f"{len(graph_record_ids)} graph documents selected from "
                    f"{graph_trace.graph_candidate_count} candidates"
                )
                span.set_outputs(
                    {
                        "selected_document_count": len(graph_record_ids),
                        "selected_chunk_count": len(graph_hits),
                        "graph_candidate_count": graph_trace.graph_candidate_count,
                    }
                )
                trace.append(
                    ToolExecution(
                        tool_name="graph_expand",
                        status="ok",
                        result_count=len(graph_hits),
                        detail=detail,
                    )
                )
                return {
                    "graph_hits": graph_hits,
                    "graph_trace": graph_trace,
                    "graph_candidate_count": len(graph_record_ids),
                    "tool_trace": trace,
                    "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
                }
            except Exception as exc:  # Graph retrieval is optional; dense evidence remains usable.
                detail = f"{type(exc).__name__}: {str(exc)[:160]}"
                span.set_outputs({"status": "error", "detail": detail})
                trace.append(
                    ToolExecution(
                        tool_name="graph_expand",
                        status="error",
                        result_count=0,
                        detail=detail,
                    )
                )
                return {
                    "graph_hits": [],
                    "graph_candidate_count": 0,
                    "tool_trace": trace,
                    "tool_call_count": int(state.get("tool_call_count", 0)) + 1,
                }

    def _synthesize_node(self, state: AgentState) -> dict[str, Any]:
        dense_hits = list(state.get("dense_hits", []))
        graph_hits = list(state.get("graph_hits", []))
        evidence = self._context_builder.build_augmented(
            dense_hits=dense_hits,
            graph_hits=graph_hits,
        )
        graph_was_called = any(
            item.tool_name == "graph_expand" for item in state.get("tool_trace", [])
        )
        strategy = "agent_dense_plus_graph" if graph_was_called else "agent_dense"

        with self._tracer.span(
            "synthesize",
            span_type="LLM",
            inputs={
                "question": _trace_text_metadata(state["question"]),
                "evidence_source_count": len(evidence),
                "retrieval_strategy": strategy,
            },
            attributes={"enterprise.operation": "grounded_answer"},
        ) as span:
            answer = self._answer_service.generate_from_evidence(
                question=state["question"],
                evidence=evidence,
                retrieved_chunk_count=len(dense_hits) + len(graph_hits),
                retrieval_strategy=strategy,
                graph_candidate_count=int(state.get("graph_candidate_count", 0)),
            )
            span.set_outputs(
                {
                    "status": answer.status.value,
                    "citation_ids": [item.citation_id for item in answer.citations],
                    "citation_count": len(answer.citations),
                }
            )
            _set_llm_span_attributes(
                span,
                model_name=answer.model_name,
                provider_name=self._llm_provider_name,
                usage=answer.usage,
            )
            return {"answer": answer}
