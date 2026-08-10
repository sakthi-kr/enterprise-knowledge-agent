"""Neo4j persistence for the enterprise knowledge graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT eka_document_record_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.record_id IS UNIQUE",
    "CREATE CONSTRAINT eka_entity_entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE INDEX eka_document_doc_id IF NOT EXISTS FOR (d:Document) ON (d.doc_id)",
    "CREATE INDEX eka_document_source_type IF NOT EXISTS FOR (d:Document) ON (d.source_type)",
    "CREATE INDEX eka_entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX eka_entity_normalized_key IF NOT EXISTS FOR (e:Entity) ON (e.normalized_key)",
)


class Neo4jGraphStore:
    """Thin wrapper around the official Neo4j Python driver."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        driver: Any | None = None,
    ) -> None:
        self.database = database
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError(
                    "Neo4j support requires the graph dependencies. "
                    'Install them with: python -m pip install -e ".[dev,nlp,graph]"'
                ) from exc
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver = driver

    def close(self) -> None:
        """Close the underlying Neo4j driver."""

        self._driver.close()

    def verify_connectivity(self) -> None:
        """Verify that the configured Neo4j instance is reachable."""

        self._driver.verify_connectivity()

    def _run(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(
            query,
            parameters_=dict(parameters or {}),
            database_=self.database,
        )
        return [record.data() for record in records]

    def ensure_schema(self) -> None:
        """Create the constraints and indexes required by the graph."""

        for statement in SCHEMA_STATEMENTS:
            self._run(statement)

    def clear_knowledge_graph(self) -> None:
        """Delete only nodes owned by this project's knowledge graph."""

        self._run("MATCH (d:Document) DETACH DELETE d")
        self._run("MATCH (e:Entity) DETACH DELETE e")

    def graph_counts(self) -> dict[str, int]:
        """Return exact counts for the graph's nodes and relationships."""

        queries = {
            "document_count": "MATCH (d:Document) RETURN count(d) AS count",
            "entity_count": "MATCH (e:Entity) RETURN count(e) AS count",
            "mention_relationship_count": (
                "MATCH (:Document)-[r:MENTIONS]->(:Entity) RETURN count(r) AS count"
            ),
            "cooccurrence_relationship_count": (
                "MATCH (:Entity)-[r:CO_OCCURS_WITH]->(:Entity) RETURN count(r) AS count"
            ),
        }
        counts: dict[str, int] = {}
        for key, query in queries.items():
            rows = self._run(query)
            counts[key] = int(rows[0]["count"]) if rows else 0
        return counts

    def upsert_entities(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Create or update canonical Entity nodes."""

        if not rows:
            return
        self._run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {entity_id: row.entity_id})
            SET e.entity_type = row.entity_type,
                e.normalized_key = row.normalized_key,
                e.display_name = row.display_name,
                e.aliases = row.aliases,
                e.mention_count = row.mention_count,
                e.document_count = row.document_count,
                e.source_types = row.source_types,
                e.max_confidence = row.max_confidence
            """,
            {"rows": list(rows)},
        )

    def upsert_documents(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Create or update Document nodes with source provenance."""

        if not rows:
            return
        self._run(
            """
            UNWIND $rows AS row
            MERGE (d:Document {record_id: row.record_id})
            SET d.doc_id = row.doc_id,
                d.source_type = row.source_type,
                d.title = row.title,
                d.source_archive = row.source_archive,
                d.source_file = row.source_file,
                d.extraction_input_characters = row.extraction_input_characters,
                d.extraction_truncated = row.extraction_truncated
            """,
            {"rows": list(rows)},
        )

    def upsert_mentions(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Create evidence-backed Document-to-Entity MENTIONS relationships."""

        if not rows:
            return
        self._run(
            """
            UNWIND $rows AS row
            MATCH (d:Document {record_id: row.record_id})
            MATCH (e:Entity {entity_id: row.entity_id})
            MERGE (d)-[r:MENTIONS]->(e)
            SET r.mention_count = row.mention_count,
                r.max_confidence = row.max_confidence,
                r.aliases = row.aliases
            """,
            {"rows": list(rows)},
        )

    def increment_cooccurrences(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Create canonical Entity-to-Entity co-occurrence relationships."""

        if not rows:
            return
        self._run(
            """
            UNWIND $rows AS row
            MATCH (a:Entity {entity_id: row.source_entity_id})
            MATCH (b:Entity {entity_id: row.target_entity_id})
            MERGE (a)-[r:CO_OCCURS_WITH]->(b)
            ON CREATE SET r.document_count = row.document_count
            ON MATCH SET r.document_count = r.document_count + row.document_count
            """,
            {"rows": list(rows)},
        )

    def schema_objects(self) -> dict[str, list[str]]:
        """Return this project's Neo4j constraint and index names."""

        constraints = self._run(
            "SHOW CONSTRAINTS YIELD name WHERE name STARTS WITH 'eka_' RETURN name ORDER BY name"
        )
        indexes = self._run(
            "SHOW INDEXES YIELD name WHERE name STARTS WITH 'eka_' RETURN name ORDER BY name"
        )
        return {
            "constraints": [str(row["name"]) for row in constraints],
            "indexes": [str(row["name"]) for row in indexes],
        }

    def top_entities(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return entities mentioned by the most source documents."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        return self._run(
            """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            RETURN e.entity_id AS entity_id,
                   e.display_name AS display_name,
                   e.entity_type AS entity_type,
                   count(DISTINCT d) AS document_count
            ORDER BY document_count DESC, display_name ASC
            LIMIT $limit
            """,
            {"limit": limit},
        )

    def top_cooccurrences(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most frequently co-occurring entity pairs."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        return self._run(
            """
            MATCH (a:Entity)-[r:CO_OCCURS_WITH]->(b:Entity)
            RETURN a.entity_id AS source_entity_id,
                   a.display_name AS source_name,
                   b.entity_id AS target_entity_id,
                   b.display_name AS target_name,
                   r.document_count AS document_count
            ORDER BY document_count DESC, source_name ASC, target_name ASC
            LIMIT $limit
            """,
            {"limit": limit},
        )
