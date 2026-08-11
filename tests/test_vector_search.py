"""Tests for dense-vector retrieval."""

from typing import Any

from enterprise_knowledge_agent.vector_search import VectorRetriever


class _FakeEncoder:
    model_name = "fake"
    dimension = 3

    def embed_passages(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        assert text == "why did alpha fail?"
        return [0.1, 0.2, 0.3]


_POINT = {
    "id": "point-1",
    "score": 0.97,
    "payload": {
        "chunk_id": "chunk-1",
        "record_id": "record-1",
        "doc_id": "doc-1",
        "source_type": "jira",
        "title": "Alpha incident",
        "source_file": "jira/alpha.txt",
        "chunk_index": 2,
        "text": "The service failed after a configuration change.",
    },
}


class _FakeStore:
    def __init__(self) -> None:
        self.last_group_filter: dict[str, Any] | None = None

    def query_points(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: dict[str, Any] | None = None,
    ):
        assert collection_name == "demo"
        assert query_vector == [0.1, 0.2, 0.3]
        assert limit == 2
        assert query_filter is None
        return [_POINT]

    def query_point_groups(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        group_by: str,
        limit: int,
        group_size: int = 1,
        query_filter: dict[str, Any] | None = None,
    ):
        assert collection_name == "demo"
        assert query_vector == [0.1, 0.2, 0.3]
        assert group_size == 1
        self.last_group_filter = query_filter
        if group_by == "doc_id":
            assert limit == 2
            assert query_filter is None
            return [{"id": "doc-1", "hits": [_POINT]}]
        if group_by == "record_id":
            assert limit == 1
            return [{"id": "record-1", "hits": [_POINT]}]
        raise AssertionError(f"Unexpected group_by: {group_by}")


def test_vector_retriever_maps_qdrant_payload() -> None:
    retriever = VectorRetriever(
        store=_FakeStore(),
        encoder=_FakeEncoder(),
        collection_name="demo",
    )

    hits = retriever.search("why did alpha fail?", limit=2)

    assert len(hits) == 1
    assert hits[0].rank == 1
    assert hits[0].score == 0.97
    assert hits[0].doc_id == "doc-1"
    assert hits[0].source_type == "jira"


def test_vector_retriever_returns_one_hit_per_document_group() -> None:
    retriever = VectorRetriever(
        store=_FakeStore(),
        encoder=_FakeEncoder(),
        collection_name="demo",
    )

    hits = retriever.search_documents("why did alpha fail?", limit=2)

    assert len(hits) == 1
    assert hits[0].rank == 1
    assert hits[0].doc_id == "doc-1"


def test_vector_retriever_filters_selected_record_ids() -> None:
    store = _FakeStore()
    retriever = VectorRetriever(
        store=store,
        encoder=_FakeEncoder(),
        collection_name="demo",
    )

    hits = retriever.search_records(
        "why did alpha fail?",
        record_ids=["record-1", "record-1"],
    )

    assert [hit.record_id for hit in hits] == ["record-1"]
    assert store.last_group_filter == {
        "must": [{"key": "record_id", "match": {"any": ["record-1"]}}]
    }
