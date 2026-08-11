"""Tests for the agent question API."""

from fastapi.testclient import TestClient

from enterprise_knowledge_agent.agent_types import (
    AgentPlan,
    AgentResult,
    AgentStrategy,
    ToolExecution,
)
from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    EvidenceSource,
    GroundedAnswer,
    TokenUsage,
)
from enterprise_knowledge_agent.main import app, provide_agent_service


class _FakeAgentService:
    def run(self, question: str) -> AgentResult:
        assert question == "How are the API gateway and autoscaler related?"
        source = EvidenceSource(
            citation_id="S1",
            rank=1,
            score=0.91,
            chunk_id="chunk-1",
            record_id="record-1",
            doc_id="doc-1",
            source_type="jira",
            title="Incident",
            source_file="incident.txt",
            text="The API gateway and autoscaler were involved in the incident.",
            retrieval_source="dense",
        )
        answer = GroundedAnswer(
            status=AnswerStatus.ANSWERED,
            answer="The components were connected through the incident [S1].",
            citations=(source,),
            model_name="answer-model",
            usage=TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
            retrieved_chunk_count=12,
            context_source_count=6,
            retrieval_strategy="agent_dense_plus_graph",
            graph_context_source_count=0,
            graph_candidate_count=2,
        )
        return AgentResult(
            answer=answer,
            plan=AgentPlan(
                strategy=AgentStrategy.DENSE_PLUS_GRAPH,
                reason="Relationship question benefits from graph expansion.",
                model_name="planner-model",
                usage=TokenUsage(prompt_tokens=8, output_tokens=3, total_tokens=11),
            ),
            tool_trace=(
                ToolExecution("dense_search", "ok", 12, "Qdrant retrieval"),
                ToolExecution("graph_expand", "ok", 0, "2 graph candidates"),
            ),
            planner_fallback=False,
            tool_call_count=2,
        )


def test_agent_ask_endpoint_returns_plan_tools_and_grounded_answer() -> None:
    app.dependency_overrides[provide_agent_service] = lambda: _FakeAgentService()
    try:
        client = TestClient(app)
        response = client.post(
            "/agent/ask",
            json={"question": "How are the API gateway and autoscaler related?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["strategy"] == "dense_plus_graph"
    assert payload["plan"]["usage"]["total_tokens"] == 11
    assert payload["tool_call_count"] == 2
    assert [item["tool_name"] for item in payload["tool_trace"]] == [
        "dense_search",
        "graph_expand",
    ]
    assert payload["answer"]["status"] == "answered"
    assert payload["answer"]["citations"][0]["citation_id"] == "S1"
