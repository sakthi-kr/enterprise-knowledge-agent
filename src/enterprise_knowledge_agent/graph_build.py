"""Build the Neo4j enterprise knowledge graph from extracted entity records."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.data_pipeline import read_jsonl
from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore

DEFAULT_ENTITIES_FILE = Path("data/processed/enterprise_rag_bench/entities/entities.jsonl")
DEFAULT_MENTIONS_FILE = Path("data/processed/enterprise_rag_bench/entities/entity_mentions.jsonl")
DEFAULT_ARTIFACT_FILE = Path("artifacts/graph/graph_build_stats.json")


class GraphStore(Protocol):
    """Operations required by the graph builder."""

    def verify_connectivity(self) -> None: ...

    def ensure_schema(self) -> None: ...

    def clear_knowledge_graph(self) -> None: ...

    def graph_counts(self) -> dict[str, int]: ...

    def upsert_entities(self, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def upsert_documents(self, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def upsert_mentions(self, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def increment_cooccurrences(self, rows: Sequence[Mapping[str, Any]]) -> None: ...


class GraphInputError(ValueError):
    """Raised when generated graph input records are inconsistent."""


def _batched(records: Iterable[Mapping[str, Any]], size: int) -> Iterable[list[Mapping[str, Any]]]:
    if size <= 0:
        raise ValueError("batch size must be greater than zero")
    batch: list[Mapping[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _require_string(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise GraphInputError(f"{context} requires a non-empty string field '{field}'")
    return value


def _canonical_entity_row(record: Mapping[str, Any]) -> dict[str, Any]:
    entity_id = _require_string(record, "entity_id", context="Entity record")
    aliases = record.get("aliases", [])
    source_types = record.get("source_types", [])
    if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
        raise GraphInputError(f"Entity {entity_id} has an invalid aliases list")
    if not isinstance(source_types, list) or not all(
        isinstance(item, str) for item in source_types
    ):
        raise GraphInputError(f"Entity {entity_id} has an invalid source_types list")
    return {
        "entity_id": entity_id,
        "entity_type": _require_string(record, "entity_type", context=f"Entity {entity_id}"),
        "normalized_key": _require_string(record, "normalized_key", context=f"Entity {entity_id}"),
        "display_name": _require_string(record, "display_name", context=f"Entity {entity_id}"),
        "aliases": aliases,
        "mention_count": int(record.get("mention_count", 0)),
        "document_count": int(record.get("document_count", 0)),
        "source_types": source_types,
        "max_confidence": float(record.get("max_confidence", 0.0)),
    }


def _document_row(record: Mapping[str, Any]) -> dict[str, Any]:
    record_id = _require_string(record, "record_id", context="Mention record")
    return {
        "record_id": record_id,
        "doc_id": _require_string(record, "doc_id", context=f"Document {record_id}"),
        "source_type": _require_string(record, "source_type", context=f"Document {record_id}"),
        "title": str(record.get("title", "")),
        "source_archive": str(record.get("source_archive", "")),
        "source_file": str(record.get("source_file", "")),
        "extraction_input_characters": int(record.get("input_characters", 0)),
        "extraction_truncated": bool(record.get("truncated", False)),
    }


def _document_relationship_rows(
    record: Mapping[str, Any],
    valid_entity_ids: set[str],
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str]]]:
    record_id = _require_string(record, "record_id", context="Mention record")
    raw_entities = record.get("entities", [])
    if not isinstance(raw_entities, list):
        raise GraphInputError(f"Document {record_id} has an invalid entities list")

    grouped: dict[str, dict[str, Any]] = {}
    for mention in raw_entities:
        if not isinstance(mention, Mapping):
            continue
        entity_id = _require_string(
            mention,
            "entity_id",
            context=f"Entity mention in document {record_id}",
        )
        if entity_id not in valid_entity_ids:
            raise GraphInputError(
                f"Document {record_id} references unknown canonical entity {entity_id}"
            )
        item = grouped.setdefault(
            entity_id,
            {
                "record_id": record_id,
                "entity_id": entity_id,
                "aliases": set(),
                "mention_count": 0,
                "max_confidence": 0.0,
            },
        )
        text = mention.get("text")
        if isinstance(text, str) and text:
            item["aliases"].add(text)
        item["mention_count"] += 1
        item["max_confidence"] = max(
            float(item["max_confidence"]),
            float(mention.get("confidence", 0.0)),
        )

    mention_rows = [
        {
            "record_id": item["record_id"],
            "entity_id": item["entity_id"],
            "aliases": sorted(item["aliases"]),
            "mention_count": int(item["mention_count"]),
            "max_confidence": round(float(item["max_confidence"]), 6),
        }
        for item in grouped.values()
    ]
    mention_rows.sort(key=lambda row: row["entity_id"])

    entity_ids = sorted(grouped)
    pairs: Counter[tuple[str, str]] = Counter()
    for index, source_id in enumerate(entity_ids):
        for target_id in entity_ids[index + 1 :]:
            pairs[(source_id, target_id)] += 1
    return mention_rows, pairs


def _cooccurrence_rows(pairs: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "document_count": count,
        }
        for (source_id, target_id), count in sorted(pairs.items())
    ]


def build_knowledge_graph(
    *,
    entities_file: Path,
    mentions_file: Path,
    store: GraphStore,
    write_batch_size: int,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Build the graph and verify the main node and relationship counts."""

    if write_batch_size <= 0:
        raise ValueError("write_batch_size must be greater than zero")
    if not entities_file.is_file():
        raise FileNotFoundError(f"Canonical entities file not found: {entities_file}")
    if not mentions_file.is_file():
        raise FileNotFoundError(f"Entity mentions file not found: {mentions_file}")

    started = time.perf_counter()
    store.verify_connectivity()
    store.ensure_schema()
    existing_counts = store.graph_counts()
    if any(existing_counts.values()):
        if not replace_existing:
            raise RuntimeError(
                "Knowledge graph already contains data. Rebuild only when replacement is intended."
            )
        store.clear_knowledge_graph()

    canonical_ids: set[str] = set()
    entity_count = 0
    for batch in _batched(read_jsonl(entities_file), write_batch_size):
        rows = [_canonical_entity_row(record) for record in batch]
        for row in rows:
            entity_id = str(row["entity_id"])
            if entity_id in canonical_ids:
                raise GraphInputError(f"Duplicate canonical entity ID: {entity_id}")
            canonical_ids.add(entity_id)
        store.upsert_entities(rows)
        entity_count += len(rows)

    document_count = 0
    documents_with_entities = 0
    mention_relationship_count = 0
    cooccurrence_pair_occurrences = 0
    source_document_counts: Counter[str] = Counter()

    for batch in _batched(read_jsonl(mentions_file), write_batch_size):
        document_rows: list[dict[str, Any]] = []
        mention_rows: list[dict[str, Any]] = []
        batch_pairs: Counter[tuple[str, str]] = Counter()

        for record in batch:
            document = _document_row(record)
            document_rows.append(document)
            source_document_counts[document["source_type"]] += 1

            relationships, pairs = _document_relationship_rows(record, canonical_ids)
            if relationships:
                documents_with_entities += 1
            mention_rows.extend(relationships)
            batch_pairs.update(pairs)

        store.upsert_documents(document_rows)
        for mention_batch in _batched(mention_rows, write_batch_size * 4):
            store.upsert_mentions(mention_batch)
        pair_rows = _cooccurrence_rows(batch_pairs)
        for pair_batch in _batched(pair_rows, write_batch_size * 4):
            store.increment_cooccurrences(pair_batch)

        document_count += len(document_rows)
        mention_relationship_count += len(mention_rows)
        cooccurrence_pair_occurrences += sum(batch_pairs.values())
        print(f"Loaded {document_count} documents into Neo4j...", flush=True)

    graph_counts = store.graph_counts()
    if graph_counts["document_count"] != document_count:
        raise RuntimeError(
            "Neo4j document count does not match the generated document records: "
            f"{graph_counts['document_count']} != {document_count}"
        )
    if graph_counts["entity_count"] != entity_count:
        raise RuntimeError(
            "Neo4j entity count does not match canonical entity records: "
            f"{graph_counts['entity_count']} != {entity_count}"
        )
    if graph_counts["mention_relationship_count"] != mention_relationship_count:
        raise RuntimeError(
            "Neo4j MENTIONS count does not match document/entity pairs: "
            f"{graph_counts['mention_relationship_count']} != {mention_relationship_count}"
        )

    return {
        "entity_input_count": entity_count,
        "document_input_count": document_count,
        "documents_with_entities": documents_with_entities,
        "source_document_counts": dict(sorted(source_document_counts.items())),
        "mention_relationship_input_count": mention_relationship_count,
        "cooccurrence_pair_occurrences": cooccurrence_pair_occurrences,
        "write_batch_size": write_batch_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **graph_counts,
    }


def _write_stats(path: Path, stats: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(stats), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities-file", type=Path, default=DEFAULT_ENTITIES_FILE)
    parser.add_argument("--mentions-file", type=Path, default=DEFAULT_MENTIONS_FILE)
    parser.add_argument("--artifact-file", type=Path, default=DEFAULT_ARTIFACT_FILE)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        stats = build_knowledge_graph(
            entities_file=args.entities_file,
            mentions_file=args.mentions_file,
            store=store,
            write_batch_size=settings.graph_write_batch_size,
            replace_existing=args.replace_existing,
        )
    finally:
        store.close()

    _write_stats(args.artifact_file, stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
