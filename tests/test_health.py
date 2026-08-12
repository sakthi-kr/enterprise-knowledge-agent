"""API liveness and readiness checks."""

from fastapi.testclient import TestClient

from enterprise_knowledge_agent.config import Settings
from enterprise_knowledge_agent.main import REQUEST_ID_HEADER, create_app
from enterprise_knowledge_agent.readiness import DependencyReadiness, ReadinessReport


def _ready_report(_: Settings) -> ReadinessReport:
    return ReadinessReport(
        ready=True,
        dependencies={
            "configuration": DependencyReadiness("ok", "runtime configured"),
            "qdrant": DependencyReadiness("ok", "collection available"),
            "neo4j": DependencyReadiness("ok", "service available"),
        },
    )


def test_health_endpoint_does_not_run_dependency_checks() -> None:
    def fail_if_called(_: Settings) -> ReadinessReport:
        raise AssertionError("health must not run readiness checks")

    application = create_app(
        settings=Settings(_env_file=None),
        readiness_checker=fail_if_called,
    )
    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Enterprise Knowledge Agent",
        "environment": "development",
    }
    assert response.headers[REQUEST_ID_HEADER]


def test_ready_endpoint_returns_dependency_state() -> None:
    application = create_app(
        settings=Settings(_env_file=None, gemini_api_key="test-key"),
        readiness_checker=_ready_report,
    )
    with TestClient(application) as client:
        response = client.get("/ready", headers={REQUEST_ID_HEADER: "request-123"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "request-123"
    assert response.json() == {
        "status": "ready",
        "service": "Enterprise Knowledge Agent",
        "environment": "development",
        "dependencies": {
            "configuration": {"status": "ok", "detail": "runtime configured"},
            "qdrant": {"status": "ok", "detail": "collection available"},
            "neo4j": {"status": "ok", "detail": "service available"},
        },
    }


def test_ready_endpoint_returns_503_when_dependency_is_unavailable() -> None:
    def not_ready(_: Settings) -> ReadinessReport:
        return ReadinessReport(
            ready=False,
            dependencies={
                "configuration": DependencyReadiness("ok", "runtime configured"),
                "qdrant": DependencyReadiness("error", "service unavailable"),
                "neo4j": DependencyReadiness("ok", "service available"),
            },
        )

    application = create_app(
        settings=Settings(_env_file=None, gemini_api_key="test-key"),
        readiness_checker=not_ready,
    )
    with TestClient(application) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["dependencies"]["qdrant"]["status"] == "error"
