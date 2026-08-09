"""Tests for the Gemini REST adapter."""

import json

import httpx

from enterprise_knowledge_agent.gemini_client import GeminiAPIError, GeminiRestClient
from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    ContextBuilder,
    EvidenceSource,
    GroundedAnswerService,
)
from enterprise_knowledge_agent.vector_search import RetrievalHit


def _evidence() -> list[EvidenceSource]:
    return [
        EvidenceSource(
            citation_id="S1",
            rank=1,
            score=0.9,
            chunk_id="chunk-1",
            record_id="record-1",
            doc_id="doc-1",
            source_type="jira",
            title="Incident report",
            source_file="incident.txt",
            text="The API gateway failed after the autoscaler entered a crash loop.",
        )
    ]


def test_gemini_client_sends_structured_grounding_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("x-goog-api-key")
        captured["body"] = json.loads(request.content)
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
                                            "status": "answered",
                                            "answer": (
                                                "The autoscaler crash loop caused the failure [S1]."
                                            ),
                                        }
                                    )
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 120,
                    "candidatesTokenCount": 18,
                    "thoughtsTokenCount": 0,
                    "totalTokenCount": 138,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = GeminiRestClient(
        api_key="secret-test-key",
        model_name="gemini-3.6-flash",
        client=http_client,
    )

    result = client.generate(question="What caused the failure?", evidence=_evidence())

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer.endswith("[S1].")
    assert result.usage.total_tokens == 138
    assert captured["api_key"] == "secret-test-key"
    assert str(captured["url"]).endswith("/v1beta/models/gemini-3.6-flash:generateContent")
    body = captured["body"]
    assert isinstance(body, dict)
    generation_config = body["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseJsonSchema"]["type"] == "object"
    assert set(generation_config["responseJsonSchema"]["required"]) == {"status", "answer"}
    assert "citation_ids" not in generation_config["responseJsonSchema"]["properties"]
    assert "responseFormat" not in generation_config
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    prompt = body["contents"][0]["parts"][0]["text"]
    assert "What caused the failure?" in prompt
    assert "[S1]" in prompt
    assert "Incident report" in prompt
    http_client.close()


def test_gemini_client_ignores_unrequested_extra_fields_in_response() -> None:
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
                                            "status": "answered",
                                            "answer": (
                                                "The failure was caused by a crash loop [S1]."
                                            ),
                                            "citation_ids": ["S1", "S2"],
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
    client = GeminiRestClient(api_key="secret-test-key", client=http_client)

    result = client.generate(question="What caused the failure?", evidence=_evidence())

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer == "The failure was caused by a crash loop [S1]."
    http_client.close()


def test_gemini_client_surfaces_api_error_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = GeminiRestClient(
        api_key="secret-test-key",
        client=http_client,
        max_retries=0,
    )

    try:
        client.generate(question="Question", evidence=_evidence())
    except GeminiAPIError as exc:
        assert "HTTP 429" in str(exc)
        assert "Rate limit exceeded" in str(exc)
    else:
        raise AssertionError("Expected GeminiAPIError")
    finally:
        http_client.close()


def test_gemini_client_rejects_non_stop_finish_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "{}"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = GeminiRestClient(api_key="secret-test-key", client=http_client)

    try:
        client.generate(question="Question", evidence=_evidence())
    except GeminiAPIError as exc:
        assert "MAX_TOKENS" in str(exc)
    else:
        raise AssertionError("Expected GeminiAPIError")
    finally:
        http_client.close()


def test_grounded_service_handles_redundant_model_citation_metadata() -> None:
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
                                            "status": "answered",
                                            "answer": "The crash loop caused the outage [S1].",
                                            "citation_ids": ["S1", "S2"],
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

    class FakeRetriever:
        def search(self, query: str, *, limit: int = 5) -> list[RetrievalHit]:
            assert query == "What caused the outage?"
            assert limit == 12
            return [
                RetrievalHit(
                    rank=1,
                    score=0.95,
                    chunk_id="chunk-1",
                    record_id="record-1",
                    doc_id="doc-1",
                    source_type="jira",
                    title="Incident report",
                    source_file="incident.txt",
                    chunk_index=0,
                    text="The autoscaler entered a crash loop and caused the outage.",
                ),
                RetrievalHit(
                    rank=2,
                    score=0.85,
                    chunk_id="chunk-2",
                    record_id="record-2",
                    doc_id="doc-2",
                    source_type="confluence",
                    title="Incident review",
                    source_file="review.txt",
                    chunk_index=0,
                    text="The review contains additional incident context.",
                ),
            ]

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    model = GeminiRestClient(api_key="secret-test-key", client=http_client)
    service = GroundedAnswerService(
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        language_model=model,
        context_builder=ContextBuilder(),
        retrieval_candidates=12,
    )

    result = service.answer("What caused the outage?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer == "The crash loop caused the outage [S1]."
    assert [citation.citation_id for citation in result.citations] == ["S1"]
    http_client.close()


def test_gemini_client_retries_transient_503_then_succeeds() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(
                503,
                json={"error": {"message": "Model is temporarily overloaded"}},
            )
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
                                            "status": "answered",
                                            "answer": "The crash loop caused the outage [S1].",
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
    client = GeminiRestClient(
        api_key="secret-test-key",
        client=http_client,
        retry_base_delay_seconds=0.01,
        sleep=delays.append,
    )

    result = client.generate(question="What caused the outage?", evidence=_evidence())

    assert result.status is AnswerStatus.ANSWERED
    assert attempts == 3
    assert len(delays) == 2
    assert 0.01 <= delays[0] <= 0.0125
    assert 0.02 <= delays[1] <= 0.025
    http_client.close()


def test_gemini_client_does_not_retry_non_transient_400() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": {"message": "Invalid request"}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = GeminiRestClient(
        api_key="secret-test-key",
        client=http_client,
        sleep=delays.append,
    )

    try:
        client.generate(question="Question", evidence=_evidence())
    except GeminiAPIError as exc:
        assert "HTTP 400" in str(exc)
        assert "Invalid request" in str(exc)
    else:
        raise AssertionError("Expected GeminiAPIError")
    finally:
        http_client.close()

    assert attempts == 1
    assert delays == []


def test_gemini_client_retries_transport_error_then_succeeds() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary network error", request=request)
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
                                            "status": "insufficient_evidence",
                                            "answer": (
                                                "The supplied evidence does not answer "
                                                "this question."
                                            ),
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
    client = GeminiRestClient(
        api_key="secret-test-key",
        client=http_client,
        retry_base_delay_seconds=0.01,
        sleep=delays.append,
    )

    result = client.generate(question="Question", evidence=_evidence())

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert attempts == 2
    assert len(delays) == 1
    http_client.close()


def test_gemini_client_surfaces_503_after_retry_budget_is_exhausted() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            503,
            json={"error": {"message": "Model is temporarily overloaded"}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = GeminiRestClient(
        api_key="secret-test-key",
        client=http_client,
        max_retries=2,
        retry_base_delay_seconds=0.01,
        sleep=delays.append,
    )

    try:
        client.generate(question="Question", evidence=_evidence())
    except GeminiAPIError as exc:
        assert "HTTP 503" in str(exc)
        assert "Model is temporarily overloaded" in str(exc)
    else:
        raise AssertionError("Expected GeminiAPIError")
    finally:
        http_client.close()

    assert attempts == 3
    assert len(delays) == 2
