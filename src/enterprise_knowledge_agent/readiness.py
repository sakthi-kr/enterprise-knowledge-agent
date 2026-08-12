"""Dependency readiness checks for the production API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from enterprise_knowledge_agent.config import Settings
from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore
from enterprise_knowledge_agent.qdrant_store import QdrantStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DependencyReadiness:
    """One dependency readiness result safe to return to API clients."""

    status: str
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregated runtime readiness state."""

    ready: bool
    dependencies: dict[str, DependencyReadiness]


def check_readiness(
    settings: Settings,
    *,
    qdrant_factory: type[QdrantStore] = QdrantStore,
    neo4j_factory: type[Neo4jGraphStore] = Neo4jGraphStore,
) -> ReadinessReport:
    """Check configuration and required stores without calling the LLM provider."""

    dependencies: dict[str, DependencyReadiness] = {}

    if settings.gemini_api_key and settings.gemini_api_key.strip():
        dependencies["configuration"] = DependencyReadiness("ok", "runtime configured")
    else:
        dependencies["configuration"] = DependencyReadiness(
            "error",
            "language-model credentials are not configured",
        )

    qdrant: QdrantStore | None = None
    try:
        qdrant = qdrant_factory(
            base_url=settings.qdrant_url,
            timeout_seconds=settings.app_readiness_timeout_seconds,
        )
        qdrant.health()
        if qdrant.collection_exists(settings.qdrant_collection):
            dependencies["qdrant"] = DependencyReadiness("ok", "collection available")
        else:
            dependencies["qdrant"] = DependencyReadiness(
                "error",
                "required collection is missing",
            )
    except Exception as exc:
        LOGGER.warning(
            "Qdrant readiness check failed",
            extra={"structured": {"dependency": "qdrant", "error_type": type(exc).__name__}},
        )
        dependencies["qdrant"] = DependencyReadiness("error", "service unavailable")
    finally:
        if qdrant is not None:
            qdrant.close()

    graph: Any | None = None
    try:
        graph = neo4j_factory(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        graph.verify_connectivity()
        dependencies["neo4j"] = DependencyReadiness("ok", "service available")
    except Exception as exc:
        LOGGER.warning(
            "Neo4j readiness check failed",
            extra={"structured": {"dependency": "neo4j", "error_type": type(exc).__name__}},
        )
        dependencies["neo4j"] = DependencyReadiness("error", "service unavailable")
    finally:
        if graph is not None:
            graph.close()

    ready = all(item.status == "ok" for item in dependencies.values())
    return ReadinessReport(ready=ready, dependencies=dependencies)
