"""Runtime external-client lifecycle tests."""

from enterprise_knowledge_agent import runtime


def test_close_runtime_services_closes_registered_resources_once() -> None:
    calls: list[str] = []
    runtime._register_resource_closers(
        lambda: calls.append("qdrant"),
        lambda: calls.append("neo4j"),
        lambda: calls.append("llm"),
    )

    runtime.close_runtime_services()
    runtime.close_runtime_services()

    assert calls == ["llm", "neo4j", "qdrant"]
