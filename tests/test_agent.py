"""Tests for bounded LangGraph agent orchestration."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from enterprise_knowledge_agent.agent import EnterpriseKnowledgeAgent
from enterprise_knowledge_agent.agent_types import AgentPlan, AgentStrategy
from enterprise_knowledge_agent.gemini_client import GeminiAPIError
from enterprise_knowledge_agent.graph_retrieval import GraphRetrievalHit, GraphRetrievalTrace
from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    GraphContextBuilder,
    GroundedAnswerService,
    LanguageModelOutput,
    TokenUsage,
)
from enterprise_knowledge_agent.observability import TraceSpan
from enterprise_knowledge_agent.vector_search import RetrievalHit

QUESTION = "What caused the incident?"


class _RecordingSpan:
    def __init__(self) -> None:
        self.inputs: Any = None
        self.outputs: Any = None
        self.attributes: dict[str, Any] = {}

    def set_inputs(self, value: Any) -> None:
        self.inputs = value

    def set_outputs(self, value: Any) -> None:
        self.outputs = value

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class _RecordingTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        inputs: Any | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        span = _RecordingSpan()
        if inputs is not None:
            span.inputs = inputs
        if attributes:
            span.attributes.update(attributes)
        self.records.append({"name": name, "span_type": span_type, "span": span})
        yield span


def _chunk(*, rank: int, record_id: str, doc_id: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=1.0 - rank * 0.01,
        chunk_id=f"chunk-{record_id}",
        record_id=record_id,
        doc_id=doc_id,
        source_type="jira",
        title=f"Title {doc_id}",
        source_file=f"{doc_id}.txt",
        chunk_index=0,
        text=text,
    )


def _graph_hit(
    *,
    rank: int,
    record_id: str,
    doc_id: str,
    matched_entity_count: int,
) -> GraphRetrievalHit:
    return GraphRetrievalHit(
        rank=rank,
        fused_score=1.0 / rank,
        record_id=record_id,
        doc_id=doc_id,
        source_type="confluence",
        title=f"Graph {doc_id}",
        dense_rank=None,
        graph_rank=rank,
        graph_score=2.0,
        matched_entity_count=matched_entity_count,
        matched_entities=("API Gateway", "Autoscaler"),
        retrieval_sources=("graph",),
    )


class _Planner:
    def __init__(self, strategy: AgentStrategy) -> None:
        self.strategy = strategy
        self.calls = 0

    def plan(self, *, question: str) -> AgentPlan:
        assert question == QUESTION
        self.calls += 1
        return AgentPlan(
            strategy=self.strategy,
            reason="Test routing decision.",
            model_name="planner-model",
            usage=TokenUsage(total_tokens=7),
        )


class _FailingPlanner:
    def plan(self, *, question: str) -> AgentPlan:
        assert question == QUESTION
        raise GeminiAPIError("planner unavailable")


class _DenseRetriever:
    def __init__(self, *, record_hits: list[RetrievalHit] | None = None) -> None:
        self.dense_hits = [
            _chunk(
                rank=index,
                record_id=f"dense-{index}",
                doc_id=f"doc-{index}",
                text=f"Dense evidence {index}",
            )
            for index in range(1, 7)
        ]
        self.record_hits = record_hits or []
        self.search_calls = 0
        self.record_calls: list[list[str]] = []

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalHit]:
        assert query == QUESTION
        self.search_calls += 1
        return self.dense_hits[:limit]

    def search_records(
        self,
        query: str,
        *,
        record_ids: list[str],
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        assert query == QUESTION
        self.record_calls.append(list(record_ids))
        allowed = set(record_ids)
        hits = [hit for hit in self.record_hits if hit.record_id in allowed]
        return hits[:limit]


class _GraphRetriever:
    def __init__(
        self,
        hits: list[GraphRetrievalHit] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.hits = hits or []
        self.fail = fail
        self.calls = 0

    def search_documents_with_trace(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[list[GraphRetrievalHit], GraphRetrievalTrace]:
        assert query == QUESTION
        self.calls += 1
        if self.fail:
            raise RuntimeError("Neo4j unavailable")
        trace = GraphRetrievalTrace(
            dense_candidate_count=30,
            seed_document_count=6,
            seed_entity_count=4,
            neighbor_entity_count=6,
            graph_candidate_count=len(self.hits),
            seed_entities=("API Gateway",),
            neighbor_entities=("Autoscaler",),
        )
        return self.hits[:limit], trace


class _LanguageModel:
    def generate(self, *, question, evidence):
        assert question == QUESTION
        graph_sources = [source for source in evidence if source.retrieval_source == "graph"]
        answer = "Dense evidence supports the answer [S1]."
        if graph_sources:
            answer = "Dense and graph evidence support the answer [S1] [S5]."
        return LanguageModelOutput(
            status=AnswerStatus.ANSWERED,
            answer=answer,
            model_name="answer-model",
            usage=TokenUsage(total_tokens=11),
        )


def _agent(
    *,
    planner,
    dense: _DenseRetriever,
    graph: _GraphRetriever,
    max_tool_calls: int = 2,
    tracer=None,
) -> EnterpriseKnowledgeAgent:
    context_builder = GraphContextBuilder(
        max_sources=6,
        dense_sources=4,
        graph_sources=2,
        max_per_document=1,
        max_context_characters=5000,
    )
    answer_service = GroundedAnswerService(
        retriever=dense,
        language_model=_LanguageModel(),
        context_builder=context_builder,
        retrieval_candidates=6,
    )
    return EnterpriseKnowledgeAgent(
        planner=planner,
        dense_retriever=dense,
        graph_retriever=graph,
        answer_service=answer_service,
        context_builder=context_builder,
        retrieval_candidates=6,
        graph_document_candidates=10,
        graph_fetch_candidates=4,
        min_graph_matched_entities=2,
        max_tool_calls=max_tool_calls,
        tracer=tracer,
    )


def test_agent_dense_only_skips_graph_tool() -> None:
    planner = _Planner(AgentStrategy.DENSE_ONLY)
    dense = _DenseRetriever()
    graph = _GraphRetriever()

    result = _agent(planner=planner, dense=dense, graph=graph).run(QUESTION)

    assert planner.calls == 1
    assert dense.search_calls == 1
    assert graph.calls == 0
    assert result.tool_call_count == 1
    assert [item.tool_name for item in result.tool_trace] == ["dense_search"]
    assert result.answer.retrieval_strategy == "agent_dense"
    assert result.answer.status is AnswerStatus.ANSWERED


def test_agent_graph_strategy_executes_graph_tool_and_adds_graph_evidence() -> None:
    graph_chunk = _chunk(
        rank=1,
        record_id="graph-strong",
        doc_id="graph-doc",
        text="Graph-only evidence",
    )
    planner = _Planner(AgentStrategy.DENSE_PLUS_GRAPH)
    dense = _DenseRetriever(record_hits=[graph_chunk])
    graph = _GraphRetriever(
        [
            _graph_hit(
                rank=1,
                record_id="graph-weak",
                doc_id="weak-doc",
                matched_entity_count=1,
            ),
            _graph_hit(
                rank=2,
                record_id="graph-strong",
                doc_id="graph-doc",
                matched_entity_count=3,
            ),
        ]
    )

    result = _agent(planner=planner, dense=dense, graph=graph).run(QUESTION)

    assert graph.calls == 1
    assert dense.record_calls == [["graph-strong"]]
    assert result.tool_call_count == 2
    assert [item.tool_name for item in result.tool_trace] == [
        "dense_search",
        "graph_expand",
    ]
    assert result.answer.retrieval_strategy == "agent_dense_plus_graph"
    assert result.answer.graph_context_source_count == 1
    assert result.answer.graph_candidate_count == 1
    assert [source.retrieval_source for source in result.answer.citations] == [
        "dense",
        "graph",
    ]


def test_agent_planner_failure_falls_back_to_dense_retrieval() -> None:
    dense = _DenseRetriever()
    graph = _GraphRetriever()

    result = _agent(planner=_FailingPlanner(), dense=dense, graph=graph).run(QUESTION)

    assert result.planner_fallback is True
    assert result.plan.strategy is AgentStrategy.DENSE_ONLY
    assert result.plan.model_name == "fallback"
    assert graph.calls == 0
    assert result.tool_call_count == 1


def test_agent_graph_failure_falls_back_to_dense_evidence() -> None:
    planner = _Planner(AgentStrategy.DENSE_PLUS_GRAPH)
    dense = _DenseRetriever()
    graph = _GraphRetriever(fail=True)

    result = _agent(planner=planner, dense=dense, graph=graph).run(QUESTION)

    assert result.answer.status is AnswerStatus.ANSWERED
    assert result.answer.graph_context_source_count == 0
    assert result.tool_call_count == 2
    assert result.tool_trace[-1].tool_name == "graph_expand"
    assert result.tool_trace[-1].status == "error"
    assert "Neo4j unavailable" in result.tool_trace[-1].detail


def test_agent_max_tool_calls_prevents_unbounded_graph_execution() -> None:
    planner = _Planner(AgentStrategy.DENSE_PLUS_GRAPH)
    dense = _DenseRetriever()
    graph = _GraphRetriever()

    result = _agent(
        planner=planner,
        dense=dense,
        graph=graph,
        max_tool_calls=1,
    ).run(QUESTION)

    assert graph.calls == 0
    assert result.tool_call_count == 1
    assert result.answer.retrieval_strategy == "agent_dense"


def test_agent_tracing_records_agent_llm_retriever_and_tool_spans() -> None:
    graph_chunk = _chunk(
        rank=1,
        record_id="graph-strong",
        doc_id="graph-doc",
        text="Graph-only evidence",
    )
    tracer = _RecordingTracer()
    agent = _agent(
        planner=_Planner(AgentStrategy.DENSE_PLUS_GRAPH),
        dense=_DenseRetriever(record_hits=[graph_chunk]),
        graph=_GraphRetriever(
            [
                _graph_hit(
                    rank=1,
                    record_id="graph-strong",
                    doc_id="graph-doc",
                    matched_entity_count=3,
                )
            ]
        ),
        tracer=tracer,
    )

    result = agent.run(QUESTION)

    assert result.answer.status is AnswerStatus.ANSWERED
    assert [record["name"] for record in tracer.records] == [
        "enterprise_agent",
        "plan",
        "dense_search",
        "graph_expand",
        "synthesize",
    ]
    assert [record["span_type"] for record in tracer.records] == [
        "AGENT",
        "LLM",
        "RETRIEVER",
        "TOOL",
        "LLM",
    ]
    dense_span = tracer.records[2]["span"]
    assert isinstance(dense_span, _RecordingSpan)
    assert len(dense_span.outputs) == 6
    assert dense_span.outputs[0]["page_content"] == ""
    assert dense_span.outputs[0]["metadata"]["chunk_id"] == "chunk-dense-1"
    assert "doc_uri" not in dense_span.outputs[0]["metadata"]
    assert dense_span.inputs == {"query": {"character_count": len(QUESTION)}, "limit": 6}

    plan_span = tracer.records[1]["span"]
    assert isinstance(plan_span, _RecordingSpan)
    assert plan_span.inputs == {"question": {"character_count": len(QUESTION)}}
    assert "reason" not in plan_span.outputs
    assert plan_span.attributes["mlflow.chat.tokenUsage"]["total_tokens"] == 7

    graph_span = tracer.records[3]["span"]
    assert isinstance(graph_span, _RecordingSpan)
    assert graph_span.inputs["question"] == {"character_count": len(QUESTION)}
    assert graph_span.outputs["selected_document_count"] == 1
    assert graph_span.outputs["selected_chunk_count"] == 1
    assert "selected_record_ids" not in graph_span.outputs
    assert "selected_chunk_ids" not in graph_span.outputs

    answer_span = tracer.records[4]["span"]
    assert isinstance(answer_span, _RecordingSpan)
    assert answer_span.inputs["question"] == {"character_count": len(QUESTION)}
    assert "answer" not in answer_span.outputs
    assert answer_span.attributes["mlflow.chat.tokenUsage"]["total_tokens"] == 11

    root_span = tracer.records[0]["span"]
    assert isinstance(root_span, _RecordingSpan)
    assert root_span.inputs == {"question": {"character_count": len(QUESTION)}}
    assert root_span.outputs["tool_call_count"] == 2
    assert root_span.outputs["status"] == "answered"

    for record in tracer.records:
        span = record["span"]
        assert QUESTION not in repr(span.inputs)
        assert QUESTION not in repr(span.outputs)
        assert "Dense evidence" not in repr(span.outputs)
        assert "Graph-only evidence" not in repr(span.outputs)
        assert "Test routing decision" not in repr(span.outputs)
