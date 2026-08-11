"""Neo4j store adapter tests."""

from __future__ import annotations

from typing import Any

import pytest

from enterprise_knowledge_agent.neo4j_store import SCHEMA_STATEMENTS, Neo4jGraphStore


class FakeRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return dict(self._data)


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any], str]] = []
        self.closed = False
        self.connectivity_verified = False

    def verify_connectivity(self) -> None:
        self.connectivity_verified = True

    def close(self) -> None:
        self.closed = True

    def execute_query(
        self,
        query: str,
        *,
        parameters_: dict[str, Any],
        database_: str,
    ) -> tuple[list[FakeRecord], None, list[str]]:
        self.queries.append((query, parameters_, database_))
        return [], None, []


def test_store_uses_injected_driver_without_optional_dependency() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(
        uri="bolt://unused:7687",
        user="neo4j",
        password="password",
        database="neo4j",
        driver=driver,
    )

    store.verify_connectivity()
    store.ensure_schema()
    store.close()

    assert driver.connectivity_verified is True
    assert driver.closed is True
    assert len(driver.queries) == len(SCHEMA_STATEMENTS)
    assert all(database == "neo4j" for _, _, database in driver.queries)
    assert any("REQUIRE d.record_id IS UNIQUE" in query for query, _, _ in driver.queries)
    assert any("REQUIRE e.entity_id IS UNIQUE" in query for query, _, _ in driver.queries)


def test_store_upsert_mentions_uses_parameterized_rows() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(
        uri="bolt://unused:7687",
        user="neo4j",
        password="password",
        driver=driver,
    )
    rows = [
        {
            "record_id": "record-a",
            "entity_id": "entity-a",
            "aliases": ["API Gateway"],
            "mention_count": 1,
            "max_confidence": 0.9,
        }
    ]

    store.upsert_mentions(rows)

    query, parameters, database = driver.queries[-1]
    assert "MERGE (d)-[r:MENTIONS]->(e)" in query
    assert parameters == {"rows": rows}
    assert database == "neo4j"


def test_store_rejects_nonpositive_sample_limits() -> None:
    store = Neo4jGraphStore(
        uri="bolt://unused:7687",
        user="neo4j",
        password="password",
        driver=FakeDriver(),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        store.top_entities(0)
    with pytest.raises(ValueError, match="greater than zero"):
        store.top_cooccurrences(0)
