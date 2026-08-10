"""Tests for local enterprise entity extraction."""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from enterprise_knowledge_agent.entity_extraction import (
    aggregate_entities,
    entity_id,
    extract_corpus_entities,
    normalize_entity_key,
    normalize_extraction_result,
    select_documents,
    truncate_for_extraction,
)


class FakeBackend:
    model_name = "fake/entity-model"

    def extract_batch(
        self,
        texts: list[str],
        entity_schema: dict[str, str],
        *,
        batch_size: int,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        assert "person" in entity_schema
        assert "Specific named people" in entity_schema["person"]
        assert batch_size > 0
        assert confidence_threshold == 0.65
        results: list[dict[str, Any]] = []
        for text in texts:
            entities: dict[str, list[dict[str, Any]]] = {}
            if "Alice" in text:
                start = text.index("Alice")
                entities.setdefault("person", []).append(
                    {"text": "Alice", "confidence": 0.96, "start": start, "end": start + 5}
                )
            if "API Gateway" in text:
                start = text.index("API Gateway")
                entities.setdefault("service", []).append(
                    {
                        "text": "API Gateway",
                        "confidence": 0.91,
                        "start": start,
                        "end": start + len("API Gateway"),
                    }
                )
            if "api-gateway" in text:
                start = text.index("api-gateway")
                entities.setdefault("service", []).append(
                    {
                        "text": "api-gateway",
                        "confidence": 0.88,
                        "start": start,
                        "end": start + len("api-gateway"),
                    }
                )
            results.append({"entities": entities})
        return results


class WrongLengthBackend(FakeBackend):
    def extract_batch(
        self,
        texts: list[str],
        entity_schema: dict[str, str],
        *,
        batch_size: int,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        return []


def _write_documents(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _document(
    record_id: str,
    doc_id: str,
    content: str,
    source_type: str = "jira",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "doc_id": doc_id,
        "source_type": source_type,
        "title": content.splitlines()[0],
        "content": content,
        "source_archive": f"{source_type}_slice_0001.zip",
        "source_file": f"{source_type}/{doc_id}_example.txt",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_select_documents_balances_sources_deterministically() -> None:
    documents = [
        _document("c1", "c1", "C1", source_type="confluence"),
        _document("c2", "c2", "C2", source_type="confluence"),
        _document("j1", "j1", "J1", source_type="jira"),
        _document("g1", "g1", "G1", source_type="github"),
        _document("j2", "j2", "J2", source_type="jira"),
        _document("g2", "g2", "G2", source_type="github"),
    ]

    selected = select_documents(documents, sample_per_source=1)

    assert [item["record_id"] for item in selected] == ["c1", "j1", "g1"]


def test_select_documents_rejects_conflicting_selection_modes() -> None:
    documents = [_document("j1", "j1", "J1")]

    with pytest.raises(ValueError, match="cannot be used together"):
        select_documents(documents, limit=1, sample_per_source=1)


def test_normalize_entity_key_merges_case_and_separators() -> None:
    assert normalize_entity_key(" API-Gateway ") == "api gateway"
    assert normalize_entity_key("api_gateway") == "api gateway"
    assert normalize_entity_key("API   Gateway") == "api gateway"


def test_entity_id_is_stable_and_type_specific() -> None:
    first = entity_id("service", "api gateway")
    repeated = entity_id("service", "api gateway")
    different_type = entity_id("project", "api gateway")

    assert first == repeated
    assert first != different_type
    assert str(uuid.UUID(first)) == first


def test_truncate_for_extraction_prefers_whitespace_boundary() -> None:
    text = "alpha beta gamma delta epsilon"

    truncated, was_truncated = truncate_for_extraction(text, 18)

    assert truncated == "alpha beta gamma"
    assert was_truncated is True
    assert truncate_for_extraction("short text", 20) == ("short text", False)


def test_normalize_extraction_result_filters_and_deduplicates() -> None:
    source_text = "Alice owns API Gateway."
    result = {
        "entities": {
            "person": [
                {"text": "Alice", "confidence": 0.95, "start": 0, "end": 5},
                {"text": "Alice", "confidence": 0.95, "start": 0, "end": 5},
            ],
            "service": [
                {"text": "API Gateway", "confidence": 0.92, "start": 11, "end": 22},
                {"text": "noise", "confidence": 0.2, "start": 0, "end": 5},
            ],
            "incident": [{"text": "admission spikes", "confidence": 0.99, "start": 0, "end": 16}],
            "unknown": [{"text": "ignored", "confidence": 0.99, "start": 0, "end": 7}],
        }
    }

    mentions = normalize_extraction_result(
        result,
        source_text=source_text,
        confidence_threshold=0.65,
    )

    assert [(item["entity_type"], item["text"]) for item in mentions] == [
        ("person", "Alice"),
        ("service", "API Gateway"),
    ]
    assert mentions[1]["normalized_key"] == "api gateway"


def test_aggregate_entities_merges_aliases_across_documents(tmp_path: Path) -> None:
    mentions_file = tmp_path / "entity_mentions.jsonl"
    canonical_id = entity_id("service", "api gateway")
    rows = [
        {
            "record_id": "record-a",
            "source_type": "jira",
            "entities": [
                {
                    "entity_id": canonical_id,
                    "entity_type": "service",
                    "text": "API Gateway",
                    "normalized_key": "api gateway",
                    "confidence": 0.91,
                }
            ],
        },
        {
            "record_id": "record-b",
            "source_type": "confluence",
            "entities": [
                {
                    "entity_id": canonical_id,
                    "entity_type": "service",
                    "text": "api-gateway",
                    "normalized_key": "api gateway",
                    "confidence": 0.88,
                },
                {
                    "entity_id": canonical_id,
                    "entity_type": "service",
                    "text": "API Gateway",
                    "normalized_key": "api gateway",
                    "confidence": 0.93,
                },
            ],
        },
    ]
    _write_documents(mentions_file, rows)
    entities_file = tmp_path / "entities.jsonl"

    stats = aggregate_entities(mentions_file, entities_file)
    entities = _read_jsonl(entities_file)

    assert stats == {
        "canonical_entity_count": 1,
        "canonical_entities_by_type": {"service": 1},
    }
    assert entities[0]["display_name"] == "API Gateway"
    assert entities[0]["aliases"] == ["API Gateway", "api-gateway"]
    assert entities[0]["mention_count"] == 3
    assert entities[0]["document_count"] == 2
    assert entities[0]["source_types"] == ["confluence", "jira"]
    assert entities[0]["max_confidence"] == 0.93


def test_extract_corpus_entities_end_to_end(tmp_path: Path) -> None:
    documents_file = tmp_path / "documents.jsonl"
    rows = [
        _document("record-a", "doc-a", "Incident\nAlice investigated the API Gateway."),
        _document(
            "record-b",
            "doc-b",
            "Runbook\nThe api-gateway rollback restored service.",
            source_type="confluence",
        ),
        _document("record-c", "doc-c", "Notes\nNo named enterprise entities here."),
    ]
    _write_documents(documents_file, rows)
    output_dir = tmp_path / "entities"

    stats = extract_corpus_entities(
        documents_file=documents_file,
        output_dir=output_dir,
        backend=FakeBackend(),
        batch_size=2,
        confidence_threshold=0.65,
        max_input_characters=1000,
    )

    mentions = _read_jsonl(output_dir / "entity_mentions.jsonl")
    entities = _read_jsonl(output_dir / "entities.jsonl")
    saved_stats = json.loads((output_dir / "entity_extraction_stats.json").read_text())

    assert len(mentions) == 3
    assert not (output_dir / "entity_mentions.partial.jsonl").exists()
    assert stats == saved_stats
    assert stats["model_name"] == "fake/entity-model"
    assert stats["processed_document_count"] == 3
    assert stats["documents_with_entities"] == 2
    assert stats["truncated_document_count"] == 0
    assert stats["entity_mention_count"] == 3
    assert stats["canonical_entity_count"] == 2
    assert stats["entity_mentions_by_type"] == {"person": 1, "service": 2}
    assert stats["source_document_counts"] == {"confluence": 1, "jira": 2}
    assert {item["entity_type"] for item in entities} == {"person", "service"}
    service = next(item for item in entities if item["entity_type"] == "service")
    assert service["document_count"] == 2
    assert service["normalized_key"] == "api gateway"


def test_extract_corpus_entities_resumes_partial_output(tmp_path: Path) -> None:
    documents_file = tmp_path / "documents.jsonl"
    rows = [
        _document("record-a", "doc-a", "Incident\nAlice investigated the API Gateway."),
        _document("record-b", "doc-b", "Runbook\nThe api-gateway rollback restored service."),
    ]
    _write_documents(documents_file, rows)
    output_dir = tmp_path / "entities"
    output_dir.mkdir()
    partial = output_dir / "entity_mentions.partial.jsonl"
    existing = {
        "record_id": "record-a",
        "doc_id": "doc-a",
        "source_type": "jira",
        "title": "Incident",
        "source_archive": "jira_slice_0001.zip",
        "source_file": "jira/doc-a_example.txt",
        "input_characters": 44,
        "truncated": False,
        "entities": [
            {
                "entity_id": entity_id("person", "alice"),
                "entity_type": "person",
                "text": "Alice",
                "normalized_key": "alice",
                "confidence": 0.96,
                "start": 9,
                "end": 14,
            }
        ],
    }
    _write_documents(partial, [existing])

    stats = extract_corpus_entities(
        documents_file=documents_file,
        output_dir=output_dir,
        backend=FakeBackend(),
        batch_size=2,
        confidence_threshold=0.65,
        max_input_characters=1000,
    )

    mentions = _read_jsonl(output_dir / "entity_mentions.jsonl")
    assert [row["record_id"] for row in mentions] == ["record-a", "record-b"]
    assert stats["processed_document_count"] == 2
    assert stats["entity_mention_count"] == 2
    assert stats["canonical_entity_count"] == 2


def test_extract_corpus_entities_rejects_backend_length_mismatch(tmp_path: Path) -> None:
    documents_file = tmp_path / "documents.jsonl"
    _write_documents(documents_file, [_document("record-a", "doc-a", "Incident\nAlice")])

    with pytest.raises(RuntimeError, match="different number of results"):
        extract_corpus_entities(
            documents_file=documents_file,
            output_dir=tmp_path / "entities",
            backend=WrongLengthBackend(),
            batch_size=2,
            confidence_threshold=0.65,
            max_input_characters=1000,
        )


def test_extract_corpus_entities_requires_documents_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Prepared documents file not found"):
        extract_corpus_entities(
            documents_file=tmp_path / "missing.jsonl",
            output_dir=tmp_path / "entities",
            backend=FakeBackend(),
            batch_size=2,
            confidence_threshold=0.65,
            max_input_characters=1000,
        )


def test_gliner2_backend_uses_local_cpu_batch_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    calls: dict[str, Any] = {}

    class FakeModel:
        def batch_extract_entities(
            self,
            texts: list[str],
            labels: dict[str, str],
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            calls["texts"] = texts
            calls["labels"] = labels
            calls["batch_kwargs"] = kwargs
            return [{"entities": {}} for _ in texts]

    class FakeGLiNER2:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: Any) -> FakeModel:
            calls["model_name"] = model_name
            calls["load_kwargs"] = kwargs
            return FakeModel()

    fake_module = types.ModuleType("gliner2")
    fake_module.GLiNER2 = FakeGLiNER2
    monkeypatch.setitem(sys.modules, "gliner2", fake_module)

    from enterprise_knowledge_agent.entity_extraction import GLiNER2Backend

    backend = GLiNER2Backend("fastino/gliner2-base-v1")
    schema = {
        "person": "Specific named people.",
        "service": "Specific named deployed services.",
    }
    results = backend.extract_batch(
        ["Alice uses API Gateway."],
        schema,
        batch_size=8,
        confidence_threshold=0.65,
    )

    assert results == [{"entities": {}}]
    assert calls["model_name"] == "fastino/gliner2-base-v1"
    assert calls["load_kwargs"] == {"map_location": "cpu"}
    assert calls["texts"] == ["Alice uses API Gateway."]
    assert calls["labels"] == schema
    assert calls["batch_kwargs"] == {
        "include_confidence": True,
        "include_spans": True,
        "batch_size": 8,
        "threshold": 0.65,
    }
