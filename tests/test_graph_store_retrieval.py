"""Neo4j graph-retrieval query tests."""

from __future__ import annotations

from typing import Any

import pytest

from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore


class FakeRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return dict(self._data)


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any], str]] = []

    def close(self) -> None:
        pass

    def execute_query(
        self,
        query: str,
        *,
        parameters_: dict[str, Any],
        database_: str,
    ) -> tuple[list[FakeRecord], None, list[str]]:
        self.queries.append((query, parameters_, database_))
        return [], None, []


def _store(driver: FakeDriver) -> Neo4jGraphStore:
    return Neo4jGraphStore(
        uri="bolt://unused:7687",
        user="neo4j",
        password="password",
        driver=driver,
    )


def test_store_graph_retrieval_queries_are_parameterized() -> None:
    driver = FakeDriver()
    store = _store(driver)

    store.seed_entities(["record-a"], limit=4, max_document_count=100)
    query, parameters, _ = driver.queries[-1]
    assert "MATCH (d:Document {record_id: record_id})-[m:MENTIONS]->(e:Entity)" in query
    assert parameters == {
        "record_ids": ["record-a"],
        "max_document_count": 100,
        "limit": 4,
    }

    store.neighboring_entities(
        ["entity-a"],
        limit=6,
        max_document_count=100,
        min_cooccurrence_documents=2,
    )
    query, parameters, _ = driver.queries[-1]
    assert "[r:CO_OCCURS_WITH]-(neighbor:Entity)" in query
    assert parameters["entity_ids"] == ["entity-a"]
    assert parameters["min_cooccurrence_documents"] == 2

    weighted = [{"entity_id": "entity-a", "weight": 0.5, "kind": "seed"}]
    store.documents_for_weighted_entities(weighted, limit=8)
    query, parameters, _ = driver.queries[-1]
    assert "sum(toFloat(weighted.weight) * m.max_confidence) AS graph_score" in query
    assert parameters == {"entities": weighted, "limit": 8}


def test_store_graph_retrieval_queries_validate_limits() -> None:
    store = _store(FakeDriver())

    with pytest.raises(ValueError, match="limit"):
        store.seed_entities(["record-a"], limit=0, max_document_count=10)
    with pytest.raises(ValueError, match="max_document_count"):
        store.seed_entities(["record-a"], limit=1, max_document_count=0)
    with pytest.raises(ValueError, match="min_cooccurrence_documents"):
        store.neighboring_entities(
            ["entity-a"],
            limit=1,
            max_document_count=10,
            min_cooccurrence_documents=0,
        )
    with pytest.raises(ValueError, match="limit"):
        store.documents_for_weighted_entities(
            [{"entity_id": "entity-a", "weight": 1.0}],
            limit=0,
        )
