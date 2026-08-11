"""Tests for Gemini agent planning."""

import json

import httpx

from enterprise_knowledge_agent.agent_types import AgentStrategy
from enterprise_knowledge_agent.gemini_client import GeminiAPIError, GeminiRestClient


def test_gemini_planner_returns_validated_strategy() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "strategy": "dense_plus_graph",
                                            "reason": "Relationship question needs graph context.",
                                        }
                                    )
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 40,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 48,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = GeminiRestClient(api_key="test-key", client=http_client)

    plan = client.plan(question="How are project A and service B connected?")

    assert plan.strategy is AgentStrategy.DENSE_PLUS_GRAPH
    assert plan.reason == "Relationship question needs graph context."
    assert plan.usage.total_tokens == 48
    body = captured["body"]
    assert isinstance(body, dict)
    schema = body["generationConfig"]["responseJsonSchema"]
    assert schema["properties"]["strategy"]["enum"] == [
        "dense_only",
        "dense_plus_graph",
    ]
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    http_client.close()


def test_gemini_planner_rejects_unknown_strategy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "strategy": "invent_tool",
                                            "reason": "Invalid strategy.",
                                        }
                                    )
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = GeminiRestClient(api_key="test-key", client=http_client)

    try:
        client.plan(question="Question")
    except GeminiAPIError as exc:
        assert "invalid strategy" in str(exc)
    else:
        raise AssertionError("Expected GeminiAPIError")
    finally:
        http_client.close()
