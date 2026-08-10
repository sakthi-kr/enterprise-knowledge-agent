"""Verify the Neo4j knowledge graph and report useful graph statistics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore

DEFAULT_OUTPUT_FILE = Path("artifacts/graph/graph_verification.json")
EXPECTED_CONSTRAINTS = {
    "eka_document_record_id",
    "eka_entity_entity_id",
}
EXPECTED_INDEXES = {
    "eka_document_doc_id",
    "eka_document_record_id",
    "eka_document_source_type",
    "eka_entity_entity_id",
    "eka_entity_normalized_key",
    "eka_entity_type",
}


def verify_graph(store: Neo4jGraphStore, *, sample_limit: int = 10) -> dict[str, Any]:
    """Verify schema and return graph counts plus representative traversals."""

    if sample_limit <= 0:
        raise ValueError("sample_limit must be greater than zero")
    store.verify_connectivity()
    counts = store.graph_counts()
    if counts["document_count"] <= 0:
        raise RuntimeError("Knowledge graph contains no Document nodes")
    if counts["entity_count"] <= 0:
        raise RuntimeError("Knowledge graph contains no Entity nodes")
    if counts["mention_relationship_count"] <= 0:
        raise RuntimeError("Knowledge graph contains no MENTIONS relationships")

    schema = store.schema_objects()
    constraint_names = set(schema["constraints"])
    index_names = set(schema["indexes"])
    missing_constraints = sorted(EXPECTED_CONSTRAINTS - constraint_names)
    missing_indexes = sorted(EXPECTED_INDEXES - index_names)
    if missing_constraints or missing_indexes:
        raise RuntimeError(
            "Neo4j schema verification failed: "
            f"missing constraints={missing_constraints}, missing indexes={missing_indexes}"
        )

    return {
        **counts,
        "constraints": schema["constraints"],
        "indexes": schema["indexes"],
        "top_entities": store.top_entities(sample_limit),
        "top_cooccurrences": store.top_cooccurrences(sample_limit),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        report = verify_graph(store, sample_limit=args.sample_limit)
    finally:
        store.close()

    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
