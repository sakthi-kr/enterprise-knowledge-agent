"""Production API middleware and error-contract tests."""

from __future__ import annotations

import asyncio
import re

from fastapi.testclient import TestClient

from enterprise_knowledge_agent.config import Settings
from enterprise_knowledge_agent.main import (
    REQUEST_ID_HEADER,
    create_app,
    provide_answer_service,
)
from enterprise_knowledge_agent.qdrant_store import QdrantStoreError
from enterprise_knowledge_agent.readiness import ReadinessReport


def _ready(_: Settings) -> ReadinessReport:
    return ReadinessReport(ready=True, dependencies={})


def test_invalid_request_id_is_replaced() -> None:
    application = create_app(
        settings=Settings(_env_file=None),
        readiness_checker=_ready,
    )
    with TestClient(application) as client:
        response = client.get("/health", headers={REQUEST_ID_HEADER: "unsafe request id"})

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id != "unsafe request id"
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)


def test_not_found_uses_stable_error_envelope() -> None:
    application = create_app(
        settings=Settings(_env_file=None),
        readiness_checker=_ready,
    )
    with TestClient(application) as client:
        response = client.get("/does-not-exist", headers={REQUEST_ID_HEADER: "missing-1"})

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Not Found",
            "request_id": "missing-1",
        }
    }


def test_request_timeout_returns_504() -> None:
    application = create_app(
        settings=Settings(_env_file=None, app_request_timeout_seconds=0.01),
        readiness_checker=_ready,
    )

    @application.get("/slow")
    async def slow() -> dict[str, str]:
        await asyncio.sleep(0.05)
        return {"status": "late"}

    with TestClient(application) as client:
        response = client.get("/slow", headers={REQUEST_ID_HEADER: "timeout-1"})

    assert response.status_code == 504
    assert response.headers[REQUEST_ID_HEADER] == "timeout-1"
    assert response.json() == {
        "error": {
            "code": "request_timeout",
            "message": "The request exceeded the configured processing timeout.",
            "request_id": "timeout-1",
        }
    }


class _UnavailableAnswerService:
    def answer(self, question: str):
        del question
        raise QdrantStoreError("private qdrant failure details")


def test_dependency_failure_does_not_leak_internal_error_details() -> None:
    application = create_app(
        settings=Settings(_env_file=None),
        readiness_checker=_ready,
    )
    application.dependency_overrides[provide_answer_service] = lambda: _UnavailableAnswerService()
    try:
        with TestClient(application) as client:
            response = client.post(
                "/ask",
                headers={REQUEST_ID_HEADER: "retrieval-1"},
                json={"question": "What happened?"},
            )
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "retrieval_unavailable",
            "message": "The retrieval service is unavailable.",
            "request_id": "retrieval-1",
        }
    }
    assert "private qdrant" not in response.text
