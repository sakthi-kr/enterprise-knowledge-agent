"""LangGraph orchestration for bounded enterprise retrieval tools."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from enterprise_knowledge_agent.agent_types import (
    AgentPlan,
    AgentResult,
    AgentStrategy,
    ToolExecution,
)
from enterprise_knowledge_agent.gemini_client import GeminiAPIError
from enterprise_knowledge_agent.graph_retrieval import GraphRAGRetriever, GraphRetrievalTrace
from enterprise_knowledge_agent.grounded_answer import (
    GraphContextBuilder,
    GroundedAnswer,
    GroundedAnswerService,
    TokenUsage,
)
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
        self._dense_retriever = dense_retriever
        self._graph_retriever = graph_retriever
        self._answer_service = answer_service
        self._context_builder = context_builder
        self._retrieval_candidates = retrieval_candidates
        self._graph_document_candidates = graph_document_candidates
        self._graph_fetch_candidates = graph_fetch_candidates
        self._min_graph_matched_entities = min_graph_matched_entities
        self._max_tool_calls = max_tool_calls
        self._workflow = self._compile_workflow()

    def run(self, question: str) -> AgentResult:
        """Execute one agent workflow and return its answer plus execution metadata."""

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

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

        return AgentResult(
            answer=answer,
            plan=plan,
            tool_trace=tuple(final_state.get("tool_trace", [])),
            planner_fallback=bool(final_state.get("planner_fallback", False)),
            tool_call_count=int(final_state.get("tool_call_count", 0)),
        )

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
        try:
            plan = self._planner.plan(question=question)
            fallback = False
        except GeminiAPIError:
            plan = AgentPlan(
                strategy=AgentStrategy.DENSE_ONLY,
                reason="Planner unavailable; dense retrieval selected as the safe fallback.",
                model_name="fallback",
                usage=TokenUsage(),
            )
            fallback = True
        return {"plan": plan, "planner_fallback": fallback}

    def _dense_search_node(self, state: AgentState) -> dict[str, Any]:
        hits = self._dense_retriever.search(
            state["question"],
            limit=self._retrieval_candidates,
        )
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
            trace.append(
                ToolExecution(
                    tool_name="graph_expand",
                    status="ok",
                    result_count=len(graph_hits),
                    detail=(
                        f"{len(graph_record_ids)} graph documents selected from "
                        f"{graph_trace.graph_candidate_count} candidates"
                    ),
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
            trace.append(
                ToolExecution(
                    tool_name="graph_expand",
                    status="error",
                    result_count=0,
                    detail=f"{type(exc).__name__}: {str(exc)[:160]}",
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
        answer = self._answer_service.generate_from_evidence(
            question=state["question"],
            evidence=evidence,
            retrieved_chunk_count=len(dense_hits) + len(graph_hits),
            retrieval_strategy=strategy,
            graph_candidate_count=int(state.get("graph_candidate_count", 0)),
        )
        return {"answer": answer}
