"""Dense-vector retrieval over the prepared enterprise corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder, TextEncoder
from enterprise_knowledge_agent.qdrant_store import QdrantStore


class SearchStore(Protocol):
    """Qdrant query operations required by the retriever."""

    def query_points(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector collection for chunk hits."""

    def query_point_groups(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        group_by: str,
        limit: int,
        group_size: int = 1,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector collection for grouped hits."""


@dataclass(frozen=True)
class RetrievalHit:
    """One retrieved chunk with its similarity score and provenance."""

    rank: int
    score: float
    chunk_id: str
    record_id: str
    doc_id: str
    source_type: str
    title: str
    source_file: str
    chunk_index: int
    text: str


def _point_to_hit(point: dict[str, Any], *, rank: int) -> RetrievalHit:
    payload = point.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Qdrant result at rank {rank} has no payload")
    try:
        return RetrievalHit(
            rank=rank,
            score=float(point["score"]),
            chunk_id=str(payload["chunk_id"]),
            record_id=str(payload["record_id"]),
            doc_id=str(payload["doc_id"]),
            source_type=str(payload["source_type"]),
            title=str(payload["title"]),
            source_file=str(payload["source_file"]),
            chunk_index=int(payload["chunk_index"]),
            text=str(payload["text"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Qdrant result at rank {rank} has an invalid payload") from exc


class VectorRetriever:
    """Embed queries and retrieve enterprise chunks or ranked documents from Qdrant."""

    def __init__(
        self,
        *,
        store: SearchStore,
        encoder: TextEncoder,
        collection_name: str,
    ) -> None:
        self._store = store
        self._encoder = encoder
        self._collection_name = collection_name

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalHit]:
        """Retrieve the top matching chunks for a non-empty query."""

        self._validate_query(query, limit=limit)
        vector = self._encoder.embed_query(query)
        points = self._store.query_points(
            collection_name=self._collection_name,
            query_vector=vector,
            limit=limit,
        )
        return [_point_to_hit(point, rank=rank) for rank, point in enumerate(points, start=1)]

    def search_documents(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        """Retrieve one highest-scoring chunk for each ranked benchmark document."""

        self._validate_query(query, limit=limit)
        vector = self._encoder.embed_query(query)
        groups = self._store.query_point_groups(
            collection_name=self._collection_name,
            query_vector=vector,
            group_by="doc_id",
            limit=limit,
            group_size=1,
        )
        return self._groups_to_hits(groups, group_field="doc_id")

    def search_records(
        self,
        query: str,
        *,
        record_ids: list[str],
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve the best query-matching chunk from selected physical documents."""

        if not record_ids:
            return []
        unique_record_ids = list(dict.fromkeys(record_id for record_id in record_ids if record_id))
        if not unique_record_ids:
            return []
        resolved_limit = limit if limit is not None else len(unique_record_ids)
        self._validate_query(query, limit=resolved_limit)
        vector = self._encoder.embed_query(query)
        groups = self._store.query_point_groups(
            collection_name=self._collection_name,
            query_vector=vector,
            group_by="record_id",
            limit=min(resolved_limit, len(unique_record_ids)),
            group_size=1,
            query_filter={
                "must": [
                    {
                        "key": "record_id",
                        "match": {"any": unique_record_ids},
                    }
                ]
            },
        )
        return self._groups_to_hits(groups, group_field="record_id")

    @staticmethod
    def _groups_to_hits(
        groups: list[dict[str, Any]],
        *,
        group_field: str,
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for rank, group in enumerate(groups, start=1):
            grouped_hits = group.get("hits")
            if not isinstance(grouped_hits, list) or not grouped_hits:
                raise RuntimeError(f"Qdrant document group at rank {rank} has no hits")
            hit = _point_to_hit(grouped_hits[0], rank=rank)
            group_id = str(group.get("id", ""))
            payload_value = str(getattr(hit, group_field))
            if group_id and group_id.lower() != payload_value.lower():
                raise RuntimeError(
                    f"Qdrant document group at rank {rank} does not match its hit payload"
                )
            hits.append(hit)
        return hits

    @staticmethod
    def _validate_query(query: str, *, limit: int) -> None:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Search the enterprise vector index.")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main() -> None:
    """Run one vector search from the command line."""

    args = build_parser().parse_args()
    settings = get_settings()
    encoder = FastEmbedTextEncoder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    with QdrantStore(
        base_url=settings.qdrant_url,
        timeout_seconds=settings.qdrant_timeout_seconds,
    ) as store:
        store.health()
        retriever = VectorRetriever(
            store=store,
            encoder=encoder,
            collection_name=settings.qdrant_collection,
        )
        hits = retriever.search(args.query, limit=args.limit)
    print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
