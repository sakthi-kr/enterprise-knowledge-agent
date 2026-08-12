"""Tests for the Groq REST language-model adapter."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from enterprise_knowledge_agent.agent_types import AgentStrategy
from enterprise_knowledge_agent.groq_client import GroqAPIError, GroqRestClient
from enterprise_knowledge_agent.grounded_answer import AnswerStatus, EvidenceSource


def _evidence() -> list[EvidenceSource]:
    return [
        EvidenceSource(
            citation_id="S1",
            rank=1,
            score=0.91,
            chunk_id="chunk-1",
            record_id="record-1",
            doc_id="doc-1",
            source_type="jira",
            title="API incident",
            source_file="incident.txt",
            text="The autoscaler warm pool regression delayed capacity readiness.",
        )
    ]


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    sleep: Callable[[float], None] = lambda _seconds: None,
    max_retries: int = 0,
) -> GroqRestClient:
    transport = httpx.MockTransport(handler)
    return GroqRestClient(
        api_key="test-key",
        client=httpx.Client(transport=transport),
        sleep=sleep,
        max_retries=max_retries,
    )


def test_groq_plan_uses_strict_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["reasoning_effort"] == "low"
        assert body["include_reasoning"] is False
        assert body["citation_options"] == "disabled"
        response_format = body["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert schema["required"] == ["strategy", "reason"]
        assert schema["additionalProperties"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "strategy": "dense_plus_graph",
                                    "reason": (
                                        "The question asks how enterprise systems are connected."
                                    ),
                                }
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 30,
                    "total_tokens": 130,
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            },
        )

    client = _client(handler)
    plan = client.plan(question="How are the API gateway and autoscaler connected?")

    assert plan.strategy is AgentStrategy.DENSE_PLUS_GRAPH
    assert plan.model_name == "openai/gpt-oss-20b"
    assert plan.usage.prompt_tokens == 100
    assert plan.usage.output_tokens == 20
    assert plan.usage.thinking_tokens == 10
    assert plan.usage.total_tokens == 130


def test_groq_grounded_answer_parses_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["include_reasoning"] is False
        assert body["response_format"]["json_schema"]["name"] == ("grounded_enterprise_answer")
        prompt = body["messages"][1]["content"]
        assert "[S1]" in prompt
        assert "warm pool regression" in prompt
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "answered",
                                    "answer": "A warm-pool regression delayed readiness [S1].",
                                }
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 250,
                    "completion_tokens": 40,
                    "total_tokens": 290,
                },
            },
        )

    client = _client(handler)
    output = client.generate(question="What caused the delay?", evidence=_evidence())

    assert output.status is AnswerStatus.ANSWERED
    assert output.answer.endswith("[S1].")
    assert output.usage.prompt_tokens == 250
    assert output.usage.output_tokens == 40
    assert output.usage.thinking_tokens == 0
    assert output.usage.total_tokens == 290


def test_groq_retries_429_using_retry_after_header() -> None:
    calls = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "2"},
                json={"error": {"message": "rate limit", "type": "rate_limit_error"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"strategy": "dense_only", "reason": "Direct lookup."}
                            )
                        },
                    }
                ],
                "usage": {},
            },
        )

    client = _client(handler, sleep=delays.append, max_retries=1)
    plan = client.plan(question="What is the deployment status?")

    assert plan.strategy is AgentStrategy.DENSE_ONLY
    assert calls == 2
    assert delays == [2.0]


def test_groq_retries_422_model_generation_error() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                422,
                json={
                    "error": {
                        "message": "structured generation failed",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"strategy": "dense_only", "reason": "Direct lookup."}
                            )
                        },
                    }
                ],
                "usage": {},
            },
        )

    client = _client(handler, max_retries=1)
    plan = client.plan(question="What is the deployment status?")

    assert plan.strategy is AgentStrategy.DENSE_ONLY
    assert calls == 2


def test_groq_non_retryable_error_preserves_http_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "invalid key", "type": "authentication_error"}},
        )

    client = _client(handler, max_retries=3)
    with pytest.raises(GroqAPIError, match=r"HTTP 401"):
        client.plan(question="What is the status?")
