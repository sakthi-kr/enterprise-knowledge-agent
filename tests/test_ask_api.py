"""Tests for the grounded question API."""

from fastapi.testclient import TestClient

from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    EvidenceSource,
    GroundedAnswer,
    TokenUsage,
)
from enterprise_knowledge_agent.main import app, provide_answer_service


class _FakeAnswerService:
    def answer(self, question: str) -> GroundedAnswer:
        assert question == "What caused the incident?"
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
            text="A timeout caused the incident.",
        )
        return GroundedAnswer(
            status=AnswerStatus.ANSWERED,
            answer="A timeout caused the incident [S1].",
            citations=(source,),
            model_name="test-model",
            usage=TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
            retrieved_chunk_count=12,
            context_source_count=6,
        )


def test_ask_endpoint_returns_grounded_answer() -> None:
    app.dependency_overrides[provide_answer_service] = lambda: _FakeAnswerService()
    try:
        client = TestClient(app)
        response = client.post("/ask", json={"question": "What caused the incident?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["answer"] == "A timeout caused the incident [S1]."
    assert payload["citations"][0]["citation_id"] == "S1"
    assert payload["model_name"] == "test-model"
    assert payload["usage"]["total_tokens"] == 15
