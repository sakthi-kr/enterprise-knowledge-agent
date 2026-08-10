"""Knowledge-graph construction tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from enterprise_knowledge_agent.graph_build import (
    GraphInputError,
    _document_relationship_rows,
    build_knowledge_graph,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _entity(entity_id: str, name: str, entity_type: str = "service") -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "normalized_key": name.casefold(),
        "display_name": name,
        "aliases": [name],
        "mention_count": 1,
        "document_count": 1,
        "source_types": ["jira"],
        "max_confidence": 0.9,
    }


def _document(record_id: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "doc_id": f"doc-{record_id}",
        "source_type": "jira",
        "title": f"Document {record_id}",
        "source_archive": "jira_slice_0001.zip",
        "source_file": f"jira/{record_id}.txt",
        "input_characters": 500,
        "truncated": False,
        "entities": entities,
    }


def _mention(entity_id: str, text: str, confidence: float = 0.9) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "entity_type": "service",
        "text": text,
        "normalized_key": text.casefold(),
        "confidence": confidence,
        "start": 0,
        "end": len(text),
    }


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.mentions: dict[tuple[str, str], dict[str, Any]] = {}
        self.cooccurrences: dict[tuple[str, str], int] = {}
        self.schema_ready = False
        self.connected = False

    def verify_connectivity(self) -> None:
        self.connected = True

    def ensure_schema(self) -> None:
        self.schema_ready = True

    def clear_knowledge_graph(self) -> None:
        self.entities.clear()
        self.documents.clear()
        self.mentions.clear()
        self.cooccurrences.clear()

    def graph_counts(self) -> dict[str, int]:
        return {
            "document_count": len(self.documents),
            "entity_count": len(self.entities),
            "mention_relationship_count": len(self.mentions),
            "cooccurrence_relationship_count": len(self.cooccurrences),
        }

    def upsert_entities(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self.entities[str(row["entity_id"])] = dict(row)

    def upsert_documents(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self.documents[str(row["record_id"])] = dict(row)

    def upsert_mentions(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            key = (str(row["record_id"]), str(row["entity_id"]))
            self.mentions[key] = dict(row)

    def increment_cooccurrences(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            key = (str(row["source_entity_id"]), str(row["target_entity_id"]))
            self.cooccurrences[key] = self.cooccurrences.get(key, 0) + int(row["document_count"])


def test_document_relationship_rows_aggregate_mentions_and_pairs() -> None:
    record = _document(
        "record-a",
        [
            _mention("entity-a", "API Gateway", 0.8),
            _mention("entity-a", "api-gateway", 0.95),
            _mention("entity-b", "Kubernetes", 0.9),
            _mention("entity-c", "PostgreSQL", 0.85),
        ],
    )

    mentions, pairs = _document_relationship_rows(
        record,
        {"entity-a", "entity-b", "entity-c"},
    )

    assert len(mentions) == 3
    api_gateway = next(row for row in mentions if row["entity_id"] == "entity-a")
    assert api_gateway["mention_count"] == 2
    assert api_gateway["aliases"] == ["API Gateway", "api-gateway"]
    assert api_gateway["max_confidence"] == 0.95
    assert pairs == {
        ("entity-a", "entity-b"): 1,
        ("entity-a", "entity-c"): 1,
        ("entity-b", "entity-c"): 1,
    }


def test_document_relationship_rows_reject_unknown_entity() -> None:
    record = _document("record-a", [_mention("missing", "Unknown")])

    with pytest.raises(GraphInputError, match="unknown canonical entity"):
        _document_relationship_rows(record, {"known"})


def test_build_knowledge_graph_end_to_end(tmp_path: Path) -> None:
    entities_file = tmp_path / "entities.jsonl"
    mentions_file = tmp_path / "entity_mentions.jsonl"
    _write_jsonl(
        entities_file,
        [
            _entity("entity-a", "API Gateway"),
            _entity("entity-b", "Kubernetes", "technology"),
            _entity("entity-c", "PostgreSQL", "technology"),
        ],
    )
    _write_jsonl(
        mentions_file,
        [
            _document(
                "record-a",
                [
                    _mention("entity-a", "API Gateway"),
                    _mention("entity-b", "Kubernetes"),
                ],
            ),
            _document(
                "record-b",
                [
                    _mention("entity-a", "API Gateway"),
                    _mention("entity-b", "Kubernetes"),
                    _mention("entity-c", "PostgreSQL"),
                ],
            ),
            _document("record-c", []),
        ],
    )
    store = InMemoryGraphStore()

    stats = build_knowledge_graph(
        entities_file=entities_file,
        mentions_file=mentions_file,
        store=store,
        write_batch_size=2,
    )

    assert store.connected is True
    assert store.schema_ready is True
    assert stats["entity_input_count"] == 3
    assert stats["document_input_count"] == 3
    assert stats["documents_with_entities"] == 2
    assert stats["mention_relationship_input_count"] == 5
    assert stats["cooccurrence_pair_occurrences"] == 4
    assert stats["document_count"] == 3
    assert stats["entity_count"] == 3
    assert stats["mention_relationship_count"] == 5
    assert stats["cooccurrence_relationship_count"] == 3
    assert store.cooccurrences[("entity-a", "entity-b")] == 2
    assert store.cooccurrences[("entity-a", "entity-c")] == 1
    assert store.cooccurrences[("entity-b", "entity-c")] == 1


def test_build_knowledge_graph_refuses_existing_data(tmp_path: Path) -> None:
    entities_file = tmp_path / "entities.jsonl"
    mentions_file = tmp_path / "entity_mentions.jsonl"
    _write_jsonl(entities_file, [_entity("entity-a", "API Gateway")])
    _write_jsonl(mentions_file, [_document("record-a", [_mention("entity-a", "API Gateway")])])
    store = InMemoryGraphStore()
    store.entities["existing"] = _entity("existing", "Existing")

    with pytest.raises(RuntimeError, match="already contains data"):
        build_knowledge_graph(
            entities_file=entities_file,
            mentions_file=mentions_file,
            store=store,
            write_batch_size=10,
        )


def test_build_knowledge_graph_replaces_existing_when_requested(tmp_path: Path) -> None:
    entities_file = tmp_path / "entities.jsonl"
    mentions_file = tmp_path / "entity_mentions.jsonl"
    _write_jsonl(entities_file, [_entity("entity-a", "API Gateway")])
    _write_jsonl(mentions_file, [_document("record-a", [_mention("entity-a", "API Gateway")])])
    store = InMemoryGraphStore()
    store.entities["existing"] = _entity("existing", "Existing")

    stats = build_knowledge_graph(
        entities_file=entities_file,
        mentions_file=mentions_file,
        store=store,
        write_batch_size=10,
        replace_existing=True,
    )

    assert set(store.entities) == {"entity-a"}
    assert stats["document_count"] == 1
    assert stats["entity_count"] == 1
    assert stats["mention_relationship_count"] == 1


def test_build_knowledge_graph_requires_input_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Canonical entities file not found"):
        build_knowledge_graph(
            entities_file=tmp_path / "missing-entities.jsonl",
            mentions_file=tmp_path / "missing-mentions.jsonl",
            store=InMemoryGraphStore(),
            write_batch_size=10,
        )


def test_build_knowledge_graph_requires_positive_batch_size(tmp_path: Path) -> None:
    entities_file = tmp_path / "entities.jsonl"
    mentions_file = tmp_path / "entity_mentions.jsonl"
    _write_jsonl(entities_file, [])
    _write_jsonl(mentions_file, [])

    with pytest.raises(ValueError, match="greater than zero"):
        build_knowledge_graph(
            entities_file=entities_file,
            mentions_file=mentions_file,
            store=InMemoryGraphStore(),
            write_batch_size=0,
        )
