"""Graph-assisted retrieval that expands dense evidence through the knowledge graph."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder
from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore
from enterprise_knowledge_agent.qdrant_store import QdrantStore
from enterprise_knowledge_agent.vector_search import RetrievalHit, VectorRetriever


class DenseDocumentRetriever(Protocol):
    """Dense document search required to seed graph expansion."""

    def search_documents(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        """Return ranked dense document hits."""


class GraphRetrievalStore(Protocol):
    """Knowledge-graph queries required by graph-assisted retrieval."""

    def seed_entities(
        self,
        record_ids: list[str],
        *,
        limit: int,
        max_document_count: int,
    ) -> list[dict[str, Any]]:
        """Return informative entities attached to dense seed documents."""

    def neighboring_entities(
        self,
        entity_ids: list[str],
        *,
        limit: int,
        max_document_count: int,
        min_cooccurrence_documents: int,
    ) -> list[dict[str, Any]]:
        """Return graph neighbors of the seed entities."""

    def documents_for_weighted_entities(
        self,
        entities: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rank documents by weighted entity evidence."""


@dataclass(frozen=True)
class GraphRetrievalHit:
    """One fused document result with dense and graph provenance."""

    rank: int
    fused_score: float
    record_id: str
    doc_id: str
    source_type: str
    title: str
    dense_rank: int | None
    graph_rank: int | None
    graph_score: float | None
    matched_entity_count: int
    matched_entities: tuple[str, ...]
    retrieval_sources: tuple[str, ...]


@dataclass(frozen=True)
class GraphRetrievalTrace:
    """Diagnostics describing how a graph-assisted query was expanded."""

    dense_candidate_count: int
    seed_document_count: int
    seed_entity_count: int
    neighbor_entity_count: int
    graph_candidate_count: int
    seed_entities: tuple[str, ...]
    neighbor_entities: tuple[str, ...]


def _positive_int(value: int, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _positive_float(value: float, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _entity_specificity(document_count: int) -> float:
    return 1.0 / math.sqrt(max(document_count, 1))


def _seed_entity_weight(row: dict[str, Any]) -> float:
    confidence = float(row.get("max_confidence", 0.0))
    document_count = int(row.get("document_count", 1))
    return 2.0 * confidence * _entity_specificity(document_count)


def _neighbor_entity_weight(row: dict[str, Any]) -> float:
    document_count = int(row.get("document_count", 1))
    support = int(row.get("max_cooccurrence_documents", 0))
    support_factor = min(max(support, 0), 5) / 5.0
    return support_factor * _entity_specificity(document_count)


class GraphRAGRetriever:
    """Fuse dense ranking with evidence-seeded knowledge-graph expansion."""

    def __init__(
        self,
        *,
        dense_retriever: DenseDocumentRetriever,
        graph_store: GraphRetrievalStore,
        dense_candidates: int = 30,
        seed_documents: int = 6,
        seed_entities: int = 16,
        neighbor_entities: int = 32,
        graph_candidates: int = 40,
        max_entity_document_count: int = 500,
        min_cooccurrence_documents: int = 2,
        rrf_k: int = 60,
        dense_weight: float = 1.1,
        graph_weight: float = 1.0,
    ) -> None:
        for value, name in (
            (dense_candidates, "dense_candidates"),
            (seed_documents, "seed_documents"),
            (seed_entities, "seed_entities"),
            (neighbor_entities, "neighbor_entities"),
            (graph_candidates, "graph_candidates"),
            (max_entity_document_count, "max_entity_document_count"),
            (min_cooccurrence_documents, "min_cooccurrence_documents"),
            (rrf_k, "rrf_k"),
        ):
            _positive_int(value, name=name)
        _positive_float(dense_weight, name="dense_weight")
        _positive_float(graph_weight, name="graph_weight")

        self._dense_retriever = dense_retriever
        self._graph_store = graph_store
        self._dense_candidates = dense_candidates
        self._seed_documents = seed_documents
        self._seed_entities = seed_entities
        self._neighbor_entities = neighbor_entities
        self._graph_candidates = graph_candidates
        self._max_entity_document_count = max_entity_document_count
        self._min_cooccurrence_documents = min_cooccurrence_documents
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._graph_weight = graph_weight

    def search_documents(self, query: str, *, limit: int = 10) -> list[GraphRetrievalHit]:
        """Return fused document ranking for one question."""

        hits, _ = self.search_documents_with_trace(query, limit=limit)
        return hits

    def search_documents_with_trace(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[list[GraphRetrievalHit], GraphRetrievalTrace]:
        """Return fused ranking plus graph-expansion diagnostics."""

        if not query.strip():
            raise ValueError("query must not be empty")
        _positive_int(limit, name="limit")

        dense_limit = max(limit, self._dense_candidates)
        dense_hits = self._dense_retriever.search_documents(query, limit=dense_limit)
        seed_record_ids = self._unique_record_ids(dense_hits[: self._seed_documents])

        seed_rows: list[dict[str, Any]] = []
        neighbor_rows: list[dict[str, Any]] = []
        graph_rows: list[dict[str, Any]] = []
        if seed_record_ids:
            seed_rows = self._graph_store.seed_entities(
                seed_record_ids,
                limit=self._seed_entities,
                max_document_count=self._max_entity_document_count,
            )
            seed_entity_ids = [str(row["entity_id"]) for row in seed_rows]
            if seed_entity_ids:
                neighbor_rows = self._graph_store.neighboring_entities(
                    seed_entity_ids,
                    limit=self._neighbor_entities,
                    max_document_count=self._max_entity_document_count,
                    min_cooccurrence_documents=self._min_cooccurrence_documents,
                )
                weighted_entities = self._weighted_entities(seed_rows, neighbor_rows)
                graph_rows = self._graph_store.documents_for_weighted_entities(
                    weighted_entities,
                    limit=self._graph_candidates,
                )

        fused = self._fuse_rankings(dense_hits, graph_rows, limit=limit)
        trace = GraphRetrievalTrace(
            dense_candidate_count=len(dense_hits),
            seed_document_count=len(seed_record_ids),
            seed_entity_count=len(seed_rows),
            neighbor_entity_count=len(neighbor_rows),
            graph_candidate_count=len(graph_rows),
            seed_entities=tuple(str(row.get("display_name", "")) for row in seed_rows),
            neighbor_entities=tuple(str(row.get("display_name", "")) for row in neighbor_rows),
        )
        return fused, trace

    @staticmethod
    def _unique_record_ids(hits: list[RetrievalHit]) -> list[str]:
        seen: set[str] = set()
        record_ids: list[str] = []
        for hit in hits:
            if hit.record_id not in seen:
                seen.add(hit.record_id)
                record_ids.append(hit.record_id)
        return record_ids

    @staticmethod
    def _weighted_entities(
        seed_rows: list[dict[str, Any]],
        neighbor_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        weighted: dict[str, dict[str, Any]] = {}
        for row in seed_rows:
            entity_id = str(row["entity_id"])
            weighted[entity_id] = {
                "entity_id": entity_id,
                "weight": _seed_entity_weight(row),
                "kind": "seed",
            }
        for row in neighbor_rows:
            entity_id = str(row["entity_id"])
            if entity_id in weighted:
                continue
            weighted[entity_id] = {
                "entity_id": entity_id,
                "weight": _neighbor_entity_weight(row),
                "kind": "neighbor",
            }
        return list(weighted.values())

    def _fuse_rankings(
        self,
        dense_hits: list[RetrievalHit],
        graph_rows: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[GraphRetrievalHit]:
        candidates: dict[str, dict[str, Any]] = {}

        for dense_rank, hit in enumerate(dense_hits, start=1):
            key = hit.doc_id.lower()
            row = candidates.setdefault(
                key,
                {
                    "record_id": hit.record_id,
                    "doc_id": hit.doc_id,
                    "source_type": hit.source_type,
                    "title": hit.title,
                    "dense_rank": None,
                    "graph_rank": None,
                    "graph_score": None,
                    "matched_entity_count": 0,
                    "matched_entities": (),
                    "score": 0.0,
                },
            )
            if row["dense_rank"] is None:
                row["dense_rank"] = dense_rank
                row["score"] += self._dense_weight / (self._rrf_k + dense_rank)

        for graph_rank, graph_row in enumerate(graph_rows, start=1):
            doc_id = str(graph_row["doc_id"])
            key = doc_id.lower()
            row = candidates.setdefault(
                key,
                {
                    "record_id": str(graph_row["record_id"]),
                    "doc_id": doc_id,
                    "source_type": str(graph_row["source_type"]),
                    "title": str(graph_row["title"]),
                    "dense_rank": None,
                    "graph_rank": None,
                    "graph_score": None,
                    "matched_entity_count": 0,
                    "matched_entities": (),
                    "score": 0.0,
                },
            )
            if row["graph_rank"] is None:
                row["graph_rank"] = graph_rank
                row["graph_score"] = float(graph_row["graph_score"])
                row["matched_entity_count"] = int(graph_row["matched_entity_count"])
                row["matched_entities"] = tuple(
                    str(name) for name in graph_row.get("matched_entities", [])
                )
                row["score"] += self._graph_weight / (self._rrf_k + graph_rank)

        ordered = sorted(
            candidates.values(),
            key=lambda row: (
                -float(row["score"]),
                int(row["dense_rank"] or 10**9),
                int(row["graph_rank"] or 10**9),
                str(row["doc_id"]).lower(),
            ),
        )
        results: list[GraphRetrievalHit] = []
        for rank, row in enumerate(ordered[:limit], start=1):
            sources = []
            if row["dense_rank"] is not None:
                sources.append("dense")
            if row["graph_rank"] is not None:
                sources.append("graph")
            results.append(
                GraphRetrievalHit(
                    rank=rank,
                    fused_score=round(float(row["score"]), 9),
                    record_id=str(row["record_id"]),
                    doc_id=str(row["doc_id"]),
                    source_type=str(row["source_type"]),
                    title=str(row["title"]),
                    dense_rank=row["dense_rank"],
                    graph_rank=row["graph_rank"],
                    graph_score=row["graph_score"],
                    matched_entity_count=int(row["matched_entity_count"]),
                    matched_entities=tuple(row["matched_entities"]),
                    retrieval_sources=tuple(sources),
                )
            )
        return results


def _build_retriever() -> tuple[GraphRAGRetriever, QdrantStore, Neo4jGraphStore]:
    settings = get_settings()
    encoder = FastEmbedTextEncoder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    qdrant = QdrantStore(
        base_url=settings.qdrant_url,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    dense = VectorRetriever(
        store=qdrant,
        encoder=encoder,
        collection_name=settings.qdrant_collection,
    )
    retriever = GraphRAGRetriever(
        dense_retriever=dense,
        graph_store=graph,
        dense_candidates=settings.graphrag_dense_candidates,
        seed_documents=settings.graphrag_seed_documents,
        seed_entities=settings.graphrag_seed_entities,
        neighbor_entities=settings.graphrag_neighbor_entities,
        graph_candidates=settings.graphrag_graph_candidates,
        max_entity_document_count=settings.graphrag_max_entity_document_count,
        min_cooccurrence_documents=settings.graphrag_min_cooccurrence_documents,
        rrf_k=settings.graphrag_rrf_k,
        dense_weight=settings.graphrag_dense_weight,
        graph_weight=settings.graphrag_graph_weight,
    )
    return retriever, qdrant, graph


def main() -> None:
    """Run one graph-assisted retrieval query."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    retriever, qdrant, graph = _build_retriever()
    try:
        qdrant.health()
        graph.verify_connectivity()
        hits, trace = retriever.search_documents_with_trace(args.query, limit=args.limit)
    finally:
        qdrant.close()
        graph.close()

    print(
        json.dumps(
            {
                "trace": asdict(trace),
                "hits": [asdict(hit) for hit in hits],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
