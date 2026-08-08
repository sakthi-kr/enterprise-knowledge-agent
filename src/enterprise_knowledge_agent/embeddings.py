"""Dense text embedding support for retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 384


class TextEncoder(Protocol):
    """Interface used by indexing and retrieval code."""

    model_name: str
    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document passages."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""


class FastEmbedTextEncoder:
    """CPU-friendly dense encoder backed by FastEmbed."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = 64,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - package dependency guard
            raise RuntimeError(
                'FastEmbed is not installed. Run: python -m pip install -e ".[dev]"'
            ) from exc

        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages using FastEmbed's retrieval-specific passage encoder."""

        if not texts:
            return []
        vectors = [
            vector.tolist()
            for vector in self._model.passage_embed(texts, batch_size=self.batch_size)
        ]
        self._validate_vectors(vectors, expected_count=len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using FastEmbed's retrieval-specific query encoder."""

        if not text.strip():
            raise ValueError("query text must not be empty")
        vectors = list(self._model.query_embed(text, batch_size=1))
        if len(vectors) != 1:
            raise RuntimeError(f"Expected one query vector, received {len(vectors)}")
        vector = vectors[0].tolist()
        self._validate_vectors([vector], expected_count=1)
        return vector

    def _validate_vectors(self, vectors: Sequence[Sequence[float]], *, expected_count: int) -> None:
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"Embedding model returned {len(vectors)} vectors for {expected_count} texts"
            )
        invalid = [index for index, vector in enumerate(vectors) if len(vector) != self.dimension]
        if invalid:
            raise RuntimeError(
                f"Embedding dimension mismatch for vector {invalid[0]}: "
                f"expected {self.dimension}, received {len(vectors[invalid[0]])}"
            )
