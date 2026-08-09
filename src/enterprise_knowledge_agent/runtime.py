"""Lazy construction of runtime RAG components."""

from __future__ import annotations

from functools import lru_cache

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder
from enterprise_knowledge_agent.gemini_client import GeminiRestClient
from enterprise_knowledge_agent.grounded_answer import ContextBuilder, GroundedAnswerService
from enterprise_knowledge_agent.qdrant_store import QdrantStore
from enterprise_knowledge_agent.vector_search import VectorRetriever


class ServiceConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing."""


@lru_cache
def get_answer_service() -> GroundedAnswerService:
    """Build and cache the local RAG service for the API process."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise ServiceConfigurationError(
            "Gemini API key is not configured. Set EKA_GEMINI_API_KEY in .env."
        )

    encoder = FastEmbedTextEncoder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    store = QdrantStore(
        base_url=settings.qdrant_url,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    store.health()
    retriever = VectorRetriever(
        store=store,
        encoder=encoder,
        collection_name=settings.qdrant_collection,
    )
    language_model = GeminiRestClient(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        base_url=settings.gemini_base_url,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    context_builder = ContextBuilder(
        max_sources=settings.rag_context_sources,
        max_per_document=settings.rag_max_chunks_per_document,
        max_context_characters=settings.rag_max_context_characters,
    )
    return GroundedAnswerService(
        retriever=retriever,
        language_model=language_model,
        context_builder=context_builder,
        retrieval_candidates=settings.rag_retrieval_candidates,
    )
