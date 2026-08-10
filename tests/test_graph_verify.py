"""Knowledge-graph verification tests."""

from __future__ import annotations

from typing import Any

import pytest

from enterprise_knowledge_agent.graph_verify import verify_graph


class VerificationStore:
    def __init__(self, *, missing_schema: bool = False) -> None:
        self.connected = False
        self.missing_schema = missing_schema

    def verify_connectivity(self) -> None:
        self.connected = True

    def graph_counts(self) -> dict[str, int]:
        return {
            "document_count": 3,
            "entity_count": 2,
            "mention_relationship_count": 4,
            "cooccurrence_relationship_count": 1,
        }

    def schema_objects(self) -> dict[str, list[str]]:
        constraints = ["eka_document_record_id", "eka_entity_entity_id"]
        indexes = [
            "eka_document_doc_id",
            "eka_document_record_id",
            "eka_document_source_type",
            "eka_entity_entity_id",
            "eka_entity_normalized_key",
            "eka_entity_type",
        ]
        if self.missing_schema:
            indexes.remove("eka_entity_type")
        return {"constraints": constraints, "indexes": indexes}

    def top_entities(self, limit: int = 10) -> list[dict[str, Any]]:
        return [{"display_name": "API Gateway", "document_count": 2}][:limit]

    def top_cooccurrences(self, limit: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "source_name": "API Gateway",
                "target_name": "Kubernetes",
                "document_count": 2,
            }
        ][:limit]


def test_verify_graph_returns_schema_counts_and_samples() -> None:
    store = VerificationStore()

    report = verify_graph(store, sample_limit=5)

    assert store.connected is True
    assert report["document_count"] == 3
    assert report["entity_count"] == 2
    assert report["mention_relationship_count"] == 4
    assert report["cooccurrence_relationship_count"] == 1
    assert report["top_entities"][0]["display_name"] == "API Gateway"
    assert report["top_cooccurrences"][0]["target_name"] == "Kubernetes"


def test_verify_graph_rejects_missing_schema() -> None:
    with pytest.raises(RuntimeError, match="schema verification failed"):
        verify_graph(VerificationStore(missing_schema=True), sample_limit=5)


def test_verify_graph_rejects_nonpositive_sample_limit() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        verify_graph(VerificationStore(), sample_limit=0)
