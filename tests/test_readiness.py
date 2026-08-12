"""Dependency readiness checker tests."""

from __future__ import annotations

from enterprise_knowledge_agent.config import Settings
from enterprise_knowledge_agent.readiness import check_readiness


class _FakeQdrant:
    instances: list[_FakeQdrant] = []

    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.closed = False
        self.instances.append(self)

    def health(self) -> str:
        return "health check passed"

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name == "enterprise_knowledge_chunks"

    def close(self) -> None:
        self.closed = True


class _FakeNeo4j:
    instances: list[_FakeNeo4j] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def verify_connectivity(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_readiness_checks_configuration_and_closes_dependency_clients() -> None:
    _FakeQdrant.instances.clear()
    _FakeNeo4j.instances.clear()
    settings = Settings(
        _env_file=None,
        gemini_api_key="test-key",
        app_readiness_timeout_seconds=3.0,
    )

    report = check_readiness(
        settings,
        qdrant_factory=_FakeQdrant,
        neo4j_factory=_FakeNeo4j,
    )

    assert report.ready is True
    assert report.dependencies["configuration"].status == "ok"
    assert report.dependencies["qdrant"].status == "ok"
    assert report.dependencies["neo4j"].status == "ok"
    assert _FakeQdrant.instances[0].closed is True
    assert _FakeNeo4j.instances[0].closed is True


def test_readiness_reports_missing_model_credentials_without_exposing_values() -> None:
    settings = Settings(_env_file=None, gemini_api_key=None)

    report = check_readiness(
        settings,
        qdrant_factory=_FakeQdrant,
        neo4j_factory=_FakeNeo4j,
    )

    assert report.ready is False
    assert report.dependencies["configuration"].status == "error"
    assert "credentials" in report.dependencies["configuration"].detail
