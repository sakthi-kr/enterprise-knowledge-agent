"""Lazy construction of runtime RAG and agent components."""

from __future__ import annotations

from functools import lru_cache

from enterprise_knowledge_agent.agent import EnterpriseKnowledgeAgent
from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder
from enterprise_knowledge_agent.gemini_client import GeminiRestClient
from enterprise_knowledge_agent.graph_retrieval import GraphRAGRetriever
from enterprise_knowledge_agent.grounded_answer import (
    GraphAugmentedAnswerService,
    GraphContextBuilder,
    GroundedAnswerService,
)
from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore
from enterprise_knowledge_agent.observability import build_tracer
from enterprise_knowledge_agent.qdrant_store import QdrantStore
from enterprise_knowledge_agent.vector_search import VectorRetriever


class ServiceConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing."""


def _build_retrieval_components() -> tuple[
    VectorRetriever,
    GraphRAGRetriever,
    GeminiRestClient,
    GraphContextBuilder,
]:
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
    qdrant = QdrantStore(
        base_url=settings.qdrant_url,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    qdrant.health()
    dense = VectorRetriever(
        store=qdrant,
        encoder=encoder,
        collection_name=settings.qdrant_collection,
    )
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    graph_store.verify_connectivity()
    graph_retriever = GraphRAGRetriever(
        dense_retriever=dense,
        graph_store=graph_store,
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
    language_model = GeminiRestClient(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        base_url=settings.gemini_base_url,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    context_builder = GraphContextBuilder(
        max_sources=settings.rag_context_sources,
        dense_sources=settings.rag_dense_context_sources,
        graph_sources=settings.rag_graph_context_sources,
        max_per_document=settings.rag_max_chunks_per_document,
        max_context_characters=settings.rag_max_context_characters,
    )
    return dense, graph_retriever, language_model, context_builder


@lru_cache
def get_answer_service() -> GroundedAnswerService:
    """Build and cache the graph-augmented grounded answer service."""

    settings = get_settings()
    dense, graph_retriever, language_model, context_builder = _build_retrieval_components()
    return GraphAugmentedAnswerService(
        retriever=dense,
        graph_retriever=graph_retriever,
        language_model=language_model,
        context_builder=context_builder,
        retrieval_candidates=settings.rag_retrieval_candidates,
        graph_document_candidates=settings.rag_graph_document_candidates,
        graph_fetch_candidates=settings.rag_graph_fetch_candidates,
        min_graph_matched_entities=settings.rag_graph_min_matched_entities,
    )


@lru_cache
def get_agent_service() -> EnterpriseKnowledgeAgent:
    """Build and cache the LangGraph enterprise knowledge agent."""

    settings = get_settings()
    dense, graph_retriever, language_model, context_builder = _build_retrieval_components()
    answer_service = GroundedAnswerService(
        retriever=dense,
        language_model=language_model,
        context_builder=context_builder,
        retrieval_candidates=settings.rag_retrieval_candidates,
    )
    tracer = build_tracer(
        enabled=settings.mlflow_enabled,
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )
    return EnterpriseKnowledgeAgent(
        planner=language_model,
        dense_retriever=dense,
        graph_retriever=graph_retriever,
        answer_service=answer_service,
        context_builder=context_builder,
        retrieval_candidates=settings.rag_retrieval_candidates,
        graph_document_candidates=settings.rag_graph_document_candidates,
        graph_fetch_candidates=settings.rag_graph_fetch_candidates,
        min_graph_matched_entities=settings.rag_graph_min_matched_entities,
        max_tool_calls=settings.agent_max_tool_calls,
        tracer=tracer,
    )
