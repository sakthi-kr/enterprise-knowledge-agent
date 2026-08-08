"""Build a dense-vector Qdrant index from prepared enterprise chunks."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder, TextEncoder
from enterprise_knowledge_agent.qdrant_store import QdrantStore

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "record_id",
    "doc_id",
    "source_type",
    "title",
    "source_archive",
    "source_file",
    "chunk_index",
    "text",
}


class VectorStore(Protocol):
    """Storage operations required by the index builder."""

    def collection_exists(self, collection_name: str) -> bool:
        """Return whether the collection exists."""

    def create_collection(
        self, *, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        """Create a collection."""

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""

    def upsert_points(self, *, collection_name: str, points: Sequence[dict[str, Any]]) -> None:
        """Upsert a batch of points."""

    def count_points(self, collection_name: str) -> int:
        """Return the exact collection size."""


def iter_chunks(path: Path) -> Iterator[dict[str, Any]]:
    """Yield validated chunk records from the prepared JSONL corpus."""

    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}") from exc
            if not isinstance(chunk, dict):
                raise ValueError(f"Expected JSON object in {path} on line {line_number}")
            missing = REQUIRED_CHUNK_FIELDS.difference(chunk)
            if missing:
                raise ValueError(
                    f"Chunk on line {line_number} is missing fields: {sorted(missing)}"
                )
            if not str(chunk["text"]).strip():
                raise ValueError(f"Chunk on line {line_number} has empty text")
            yield chunk


def batched(items: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Yield fixed-size lists from an iterator without loading the corpus into memory."""

    if size <= 0:
        raise ValueError("batch size must be greater than zero")
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def chunk_to_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build the Qdrant payload stored alongside one chunk vector."""

    return {
        "chunk_id": chunk["chunk_id"],
        "record_id": chunk["record_id"],
        "doc_id": chunk["doc_id"],
        "source_type": chunk["source_type"],
        "title": chunk["title"],
        "source_archive": chunk["source_archive"],
        "source_file": chunk["source_file"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
    }


def build_vector_index(
    *,
    chunks_path: Path,
    stats_path: Path,
    store: VectorStore,
    encoder: TextEncoder,
    collection_name: str,
    upload_batch_size: int = 64,
    recreate: bool = False,
) -> dict[str, Any]:
    """Embed all chunks and upload them to a fresh Qdrant collection."""

    if upload_batch_size <= 0:
        raise ValueError("upload_batch_size must be greater than zero")

    if store.collection_exists(collection_name):
        if not recreate:
            raise RuntimeError(
                f"Collection '{collection_name}' already exists. "
                "Use --recreate to replace it deliberately."
            )
        store.delete_collection(collection_name)

    store.create_collection(
        collection_name=collection_name,
        vector_size=encoder.dimension,
        distance="Cosine",
    )

    started = time.perf_counter()
    indexed_count = 0
    batch_count = 0

    for batch in batched(iter_chunks(chunks_path), upload_batch_size):
        vectors = encoder.embed_passages([str(chunk["text"]) for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Encoder returned {len(vectors)} vectors for a batch of {len(batch)} chunks"
            )
        points = [
            {
                "id": str(chunk["chunk_id"]),
                "vector": vector,
                "payload": chunk_to_payload(chunk),
            }
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        store.upsert_points(collection_name=collection_name, points=points)
        indexed_count += len(points)
        batch_count += 1
        if batch_count == 1 or batch_count % 25 == 0:
            print(f"Indexed {indexed_count} chunks...")

    stored_count = store.count_points(collection_name)
    if stored_count != indexed_count:
        raise RuntimeError(
            f"Qdrant contains {stored_count} points after indexing {indexed_count} chunks"
        )

    stats = {
        "collection_name": collection_name,
        "distance": "Cosine",
        "embedding_dimension": encoder.dimension,
        "embedding_model": encoder.model_name,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "indexed_chunk_count": indexed_count,
        "stored_point_count": stored_count,
        "upload_batch_size": upload_batch_size,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Build the dense-vector retrieval index.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/enterprise_rag_bench/chunks.jsonl"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("artifacts/retrieval/vector_index_stats.json"),
    )
    parser.add_argument("--upload-batch-size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true")
    return parser


def main() -> None:
    """Build the configured vector index from the command line."""

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
        stats = build_vector_index(
            chunks_path=args.chunks,
            stats_path=args.stats,
            store=store,
            encoder=encoder,
            collection_name=settings.qdrant_collection,
            upload_batch_size=args.upload_batch_size,
            recreate=args.recreate,
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
