"""Tests for dense-vector index construction."""

import json
from pathlib import Path

from enterprise_knowledge_agent.vector_index import build_vector_index, iter_chunks


class _FakeEncoder:
    model_name = "fake-encoder"
    dimension = 3

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _FakeStore:
    def __init__(self, *, exists: bool = False) -> None:
        self.exists = exists
        self.created: list[tuple[str, int, str]] = []
        self.deleted: list[str] = []
        self.points: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(
        self, *, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        self.created.append((collection_name, vector_size, distance))
        self.exists = True

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)
        self.points = []
        self.exists = False

    def upsert_points(self, *, collection_name: str, points) -> None:
        self.points.extend(points)

    def count_points(self, collection_name: str) -> int:
        return len(self.points)


def _write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "11111111-1111-5111-8111-111111111111",
            "record_id": "aaaaaaaa-aaaa-5aaa-8aaa-aaaaaaaaaaaa",
            "doc_id": "dsid_11111111111111111111111111111111",
            "source_type": "jira",
            "title": "Alpha ticket",
            "source_archive": "jira_slice_0001.zip",
            "source_file": "jira/alpha.txt",
            "chunk_index": 0,
            "text": "alpha incident rollback",
        },
        {
            "chunk_id": "22222222-2222-5222-8222-222222222222",
            "record_id": "bbbbbbbb-bbbb-5bbb-8bbb-bbbbbbbbbbbb",
            "doc_id": "dsid_22222222222222222222222222222222",
            "source_type": "github",
            "title": "Beta change",
            "source_archive": "github_slice_0001.zip",
            "source_file": "github/beta.txt",
            "chunk_index": 0,
            "text": "beta dependency change",
        },
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_build_vector_index_end_to_end_with_fakes(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    stats_path = tmp_path / "stats.json"
    _write_chunks(chunks_path)
    store = _FakeStore()

    stats = build_vector_index(
        chunks_path=chunks_path,
        stats_path=stats_path,
        store=store,
        encoder=_FakeEncoder(),
        collection_name="demo",
        upload_batch_size=1,
    )

    assert store.created == [("demo", 3, "Cosine")]
    assert len(store.points) == 2
    assert store.points[0]["id"] == "11111111-1111-5111-8111-111111111111"
    assert store.points[0]["payload"]["doc_id"] == "dsid_11111111111111111111111111111111"
    assert stats["indexed_chunk_count"] == 2
    assert stats["stored_point_count"] == 2
    assert json.loads(stats_path.read_text(encoding="utf-8"))["embedding_model"] == "fake-encoder"


def test_build_vector_index_requires_explicit_recreate(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    try:
        build_vector_index(
            chunks_path=chunks_path,
            stats_path=tmp_path / "stats.json",
            store=_FakeStore(exists=True),
            encoder=_FakeEncoder(),
            collection_name="demo",
        )
    except RuntimeError as exc:
        assert "--recreate" in str(exc)
    else:
        raise AssertionError("Expected existing-collection RuntimeError")


def test_iter_chunks_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"chunk_id": "x", "text": "hello"}\n', encoding="utf-8")

    try:
        list(iter_chunks(path))
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("Expected missing-field ValueError")
