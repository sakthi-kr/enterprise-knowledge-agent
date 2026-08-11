"""Graph-assisted retrieval tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from enterprise_knowledge_agent.graph_retrieval import GraphRAGRetriever
from enterprise_knowledge_agent.vector_search import RetrievalHit


def _dense_hit(rank: int, doc_id: str, record_id: str) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=1.0 - (rank * 0.01),
        chunk_id=f"chunk-{rank}",
        record_id=record_id,
        doc_id=doc_id,
        source_type="jira",
        title=f"Document {doc_id}",
        source_file=f"{doc_id}.txt",
        chunk_index=0,
        text=f"Text for {doc_id}",
    )


class FakeDenseRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search_documents(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        self.calls.append((query, limit))
        return [replace(hit, rank=rank) for rank, hit in enumerate(self.hits[:limit], start=1)]


class FakeGraphStore:
    def __init__(
        self,
        *,
        seed_rows: list[dict[str, Any]] | None = None,
        neighbor_rows: list[dict[str, Any]] | None = None,
        document_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.seed_rows = seed_rows or []
        self.neighbor_rows = neighbor_rows or []
        self.document_rows = document_rows or []
        self.calls: list[tuple[str, Any]] = []

    def seed_entities(
        self,
        record_ids: list[str],
        *,
        limit: int,
        max_document_count: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(("seed", (record_ids, limit, max_document_count)))
        return list(self.seed_rows)

    def neighboring_entities(
        self,
        entity_ids: list[str],
        *,
        limit: int,
        max_document_count: int,
        min_cooccurrence_documents: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            (
                "neighbor",
                (
                    entity_ids,
                    limit,
                    max_document_count,
                    min_cooccurrence_documents,
                ),
            )
        )
        return list(self.neighbor_rows)

    def documents_for_weighted_entities(
        self,
        entities: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(("documents", (entities, limit)))
        return list(self.document_rows)


def test_graph_retrieval_falls_back_to_dense_when_seed_entities_are_absent() -> None:
    dense = FakeDenseRetriever(
        [_dense_hit(1, "doc-a", "record-a"), _dense_hit(2, "doc-b", "record-b")]
    )
    graph = FakeGraphStore()
    retriever = GraphRAGRetriever(
        dense_retriever=dense,
        graph_store=graph,
        dense_candidates=2,
        seed_documents=2,
    )

    hits, trace = retriever.search_documents_with_trace("gateway failure", limit=2)

    assert [hit.doc_id for hit in hits] == ["doc-a", "doc-b"]
    assert all(hit.retrieval_sources == ("dense",) for hit in hits)
    assert trace.seed_entity_count == 0
    assert trace.graph_candidate_count == 0
    assert [call[0] for call in graph.calls] == ["seed"]


def test_graph_retrieval_expands_entities_and_adds_graph_only_documents() -> None:
    dense = FakeDenseRetriever(
        [
            _dense_hit(1, "doc-a", "record-a"),
            _dense_hit(2, "doc-b", "record-b"),
            _dense_hit(3, "doc-c", "record-c"),
        ]
    )
    graph = FakeGraphStore(
        seed_rows=[
            {
                "entity_id": "entity-api",
                "display_name": "API Gateway",
                "document_count": 4,
                "max_confidence": 0.9,
            }
        ],
        neighbor_rows=[
            {
                "entity_id": "entity-redis",
                "display_name": "Redis",
                "document_count": 9,
                "max_cooccurrence_documents": 3,
            }
        ],
        document_rows=[
            {
                "record_id": "record-z",
                "doc_id": "doc-z",
                "source_type": "confluence",
                "title": "Autoscaler incident review",
                "graph_score": 0.8,
                "matched_entity_count": 2,
                "matched_entities": ["API Gateway", "Redis"],
            },
            {
                "record_id": "record-a",
                "doc_id": "doc-a",
                "source_type": "jira",
                "title": "Document doc-a",
                "graph_score": 0.7,
                "matched_entity_count": 1,
                "matched_entities": ["API Gateway"],
            },
        ],
    )
    retriever = GraphRAGRetriever(
        dense_retriever=dense,
        graph_store=graph,
        dense_candidates=3,
        seed_documents=2,
        seed_entities=4,
        neighbor_entities=4,
        graph_candidates=4,
        dense_weight=1.1,
        graph_weight=1.0,
    )

    hits, trace = retriever.search_documents_with_trace("gateway failure", limit=4)

    assert hits[0].doc_id == "doc-a"
    assert hits[0].retrieval_sources == ("dense", "graph")
    assert "doc-z" in [hit.doc_id for hit in hits]
    graph_only = next(hit for hit in hits if hit.doc_id == "doc-z")
    assert graph_only.retrieval_sources == ("graph",)
    assert graph_only.matched_entities == ("API Gateway", "Redis")
    assert trace.seed_entities == ("API Gateway",)
    assert trace.neighbor_entities == ("Redis",)

    documents_call = next(call for call in graph.calls if call[0] == "documents")
    weighted_entities = documents_call[1][0]
    assert {row["kind"] for row in weighted_entities} == {"seed", "neighbor"}
    assert all(float(row["weight"]) > 0 for row in weighted_entities)


def test_graph_retrieval_rejects_invalid_configuration_and_queries() -> None:
    dense = FakeDenseRetriever([])
    graph = FakeGraphStore()

    with pytest.raises(ValueError, match="dense_candidates"):
        GraphRAGRetriever(dense_retriever=dense, graph_store=graph, dense_candidates=0)

    retriever = GraphRAGRetriever(dense_retriever=dense, graph_store=graph)
    with pytest.raises(ValueError, match="query"):
        retriever.search_documents("   ")
    with pytest.raises(ValueError, match="limit"):
        retriever.search_documents("question", limit=0)
