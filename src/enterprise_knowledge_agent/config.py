"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    app_name: str = "Enterprise Knowledge Agent"
    app_environment: str = "development"
    log_level: str = "INFO"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "enterprise_knowledge_chunks"
    qdrant_timeout_seconds: float = 120.0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_timeout_seconds: float = 60.0
    rag_retrieval_candidates: int = 12
    rag_context_sources: int = 6
    rag_max_chunks_per_document: int = 2
    rag_max_context_characters: int = 18000
    nlp_model: str = "fastino/gliner2-base-v1"
    nlp_batch_size: int = 8
    nlp_confidence_threshold: float = 0.65
    nlp_max_input_characters: int = 1800
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "enterprise-knowledge-local"
    neo4j_database: str = "neo4j"
    graph_write_batch_size: int = 500
    graphrag_dense_candidates: int = 30
    graphrag_seed_documents: int = 6
    graphrag_seed_entities: int = 16
    graphrag_neighbor_entities: int = 32
    graphrag_graph_candidates: int = 40
    graphrag_max_entity_document_count: int = 500
    graphrag_min_cooccurrence_documents: int = 2
    graphrag_rrf_k: int = 60
    graphrag_dense_weight: float = 1.1
    graphrag_graph_weight: float = 1.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EKA_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for the application process."""

    return Settings()
