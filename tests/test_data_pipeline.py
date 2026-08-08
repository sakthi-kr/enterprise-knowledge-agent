"""Tests for EnterpriseRAG-Bench corpus preparation."""

import json
import uuid
import zipfile
from pathlib import Path

from enterprise_knowledge_agent.data_pipeline import (
    chunk_document,
    document_id_from_member,
    document_record_id,
    iter_archive_documents,
    prepare_corpus,
    source_type_from_archive,
)

DOC_A = "dsid_11111111111111111111111111111111"
DOC_B = "dsid_22222222222222222222222222222222"
DOC_C = "dsid_33333333333333333333333333333333"


def _write_archive(path: Path, documents: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in documents.items():
            archive.writestr(filename, content)


def _write_questions(path: Path) -> None:
    rows = [
        {
            "question_id": "qst_0001",
            "question_type": "Basic",
            "question": "What happened in Alpha?",
            "expected_doc_ids": [DOC_A],
            "gold_answer": "Alpha was resolved.",
        },
        {
            "question_id": "qst_0002",
            "question_type": "Project Related",
            "question": "How are Alpha and Beta connected?",
            "expected_doc_ids": [DOC_A, DOC_B],
            "gold_answer": "They share a dependency.",
        },
        {
            "question_id": "qst_0003",
            "question_type": "Semantic",
            "question": "What does Gamma describe?",
            "expected_doc_ids": [DOC_C],
            "gold_answer": "Gamma is outside this subset.",
        },
        {
            "question_id": "qst_0004",
            "question_type": "Info Not Found",
            "question": "What is Redwood's lunar office address?",
            "expected_doc_ids": [],
            "gold_answer": "The information is not available.",
        },
        {
            "question_id": "qst_0005",
            "question_type": "High Level",
            "question": "Summarize the whole company.",
            "expected_doc_ids": [],
            "gold_answer": "Requires the full corpus.",
        },
    ]
    _write_jsonl(path, rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_archive_and_document_id_parsing() -> None:
    assert source_type_from_archive(Path("github_slice_0001.zip")) == "github"
    assert document_id_from_member(f"nested/{DOC_A}_incident_report.txt") == DOC_A


def test_document_record_id_is_stable_and_source_specific() -> None:
    first = document_record_id(
        doc_id=DOC_A,
        source_type="jira",
        source_archive="jira_slice_0001.zip",
        source_file=f"jira/{DOC_A}_ticket.txt",
    )
    repeated = document_record_id(
        doc_id=DOC_A,
        source_type="jira",
        source_archive="jira_slice_0001.zip",
        source_file=f"jira/{DOC_A}_ticket.txt",
    )
    different_source = document_record_id(
        doc_id=DOC_A,
        source_type="confluence",
        source_archive="confluence_slice_0001.zip",
        source_file=f"confluence/{DOC_A}_report.txt",
    )

    assert first == repeated
    assert first != different_source
    assert str(uuid.UUID(first)) == first


def test_iter_archive_documents_handles_multiple_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "github_slice_0001.zip"
    _write_archive(
        archive_path,
        {
            f"nested/{DOC_B}_beta.txt": "Beta title\nBeta content",
            f"nested/{DOC_A}_alpha.txt": "Alpha title\nAlpha content",
        },
    )

    documents = list(iter_archive_documents(archive_path))

    assert [document["doc_id"] for document in documents] == [DOC_A, DOC_B]
    assert [document["title"] for document in documents] == ["Alpha title", "Beta title"]
    assert all(document["source_type"] == "github" for document in documents)
    assert all(document["source_archive"] == archive_path.name for document in documents)
    assert len({document["record_id"] for document in documents}) == 2


def test_chunk_document_preserves_overlap_and_provenance() -> None:
    document = {
        "record_id": "7c7c64af-22d8-5f98-a069-c399097395c8",
        "doc_id": DOC_A,
        "source_type": "confluence",
        "title": "Example",
        "content": "one two three four five six seven eight",
        "source_archive": "confluence_slice_0001.zip",
        "source_file": f"confluence/{DOC_A}_example.txt",
    }

    chunks = list(chunk_document(document, chunk_size_words=5, chunk_overlap_words=2))

    assert len(chunks) == 2
    assert chunks[0]["text"] == "one two three four five"
    assert chunks[1]["text"] == "four five six seven eight"
    assert chunks[0]["record_id"] == document["record_id"]
    assert chunks[0]["source_archive"] == document["source_archive"]
    assert chunks[0]["source_file"] == document["source_file"]
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]
    assert str(uuid.UUID(chunks[0]["chunk_id"])) == chunks[0]["chunk_id"]


def test_prepare_corpus_end_to_end(tmp_path: Path) -> None:
    archives_dir = tmp_path / "archives"
    archives_dir.mkdir()
    _write_archive(
        archives_dir / "confluence_slice_0001.zip",
        {
            f"{DOC_A}_alpha.txt": "Alpha runbook\n\nAlpha was resolved after a rollback.",
        },
    )
    _write_archive(
        archives_dir / "jira_slice_0001.zip",
        {
            f"{DOC_B}_beta.txt": "Beta ticket\n\nBeta shares a dependency with Alpha.",
        },
    )
    questions_file = tmp_path / "questions.jsonl"
    _write_questions(questions_file)
    output_dir = tmp_path / "processed"

    stats = prepare_corpus(
        archives_dir=archives_dir,
        questions_file=questions_file,
        output_dir=output_dir,
        chunk_size_words=6,
        chunk_overlap_words=2,
    )

    documents = list(_read_jsonl(output_dir / "documents.jsonl"))
    chunks = list(_read_jsonl(output_dir / "chunks.jsonl"))
    questions = list(_read_jsonl(output_dir / "benchmark_questions.jsonl"))
    saved_stats = json.loads((output_dir / "corpus_stats.json").read_text(encoding="utf-8"))

    assert [row["doc_id"] for row in documents] == [DOC_A, DOC_B]
    assert len({row["record_id"] for row in documents}) == 2
    assert len(chunks) == 4
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    assert [row["question_id"] for row in questions] == ["qst_0001", "qst_0002", "qst_0004"]
    assert stats == saved_stats
    assert stats["document_count"] == 2
    assert stats["unique_document_id_count"] == 2
    assert stats["ambiguous_document_id_count"] == 0
    assert stats["duplicate_document_record_count"] == 0
    assert stats["source_counts"] == {"confluence": 1, "jira": 1}
    assert stats["compatible_question_count"] == 3
    assert stats["skipped_questions_by_reason"] == {
        "missing_expected_documents": 1,
        "requires_full_corpus_or_has_no_gold_documents": 1,
    }


def test_prepare_corpus_preserves_conflicting_duplicate_ids(tmp_path: Path) -> None:
    archives_dir = tmp_path / "archives"
    archives_dir.mkdir()
    _write_archive(
        archives_dir / "confluence_slice_0001.zip",
        {f"{DOC_A}_incident_review.txt": "Incident review\nConfluence evidence."},
    )
    _write_archive(
        archives_dir / "jira_slice_0001.zip",
        {
            f"{DOC_A}_support_ticket.txt": "Support ticket\nDifferent Jira evidence.",
            f"{DOC_B}_old_cleanup.txt": "Old cleanup\nFirst Jira version.",
        },
    )
    _write_archive(
        archives_dir / "jira_slice_0002.zip",
        {f"{DOC_B}_new_cleanup.txt": "New cleanup\nSecond Jira version."},
    )
    questions_file = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions_file,
        [
            {
                "question_id": "qst_1001",
                "question_type": "Basic",
                "question": "What happened in the incident?",
                "expected_doc_ids": [DOC_A],
            },
            {
                "question_id": "qst_1002",
                "question_type": "Basic",
                "question": "What happened to cleanup?",
                "expected_doc_ids": [DOC_B],
            },
            {
                "question_id": "qst_1003",
                "question_type": "Info Not Found",
                "question": "What is the missing fact?",
                "expected_doc_ids": [],
            },
        ],
    )
    output_dir = tmp_path / "processed"

    stats = prepare_corpus(
        archives_dir=archives_dir,
        questions_file=questions_file,
        output_dir=output_dir,
        chunk_size_words=20,
        chunk_overlap_words=2,
    )

    documents = list(_read_jsonl(output_dir / "documents.jsonl"))
    chunks = list(_read_jsonl(output_dir / "chunks.jsonl"))
    questions = list(_read_jsonl(output_dir / "benchmark_questions.jsonl"))

    assert len(documents) == 4
    assert len({row["record_id"] for row in documents}) == 4
    assert [row["doc_id"] for row in documents].count(DOC_A) == 2
    assert [row["doc_id"] for row in documents].count(DOC_B) == 2
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    assert [row["question_id"] for row in questions] == ["qst_1003"]
    assert stats["document_count"] == 4
    assert stats["unique_document_id_count"] == 2
    assert stats["ambiguous_document_id_count"] == 2
    assert stats["ambiguous_document_ids"] == [DOC_A, DOC_B]
    assert stats["duplicate_document_record_count"] == 2
    assert stats["source_counts"] == {"confluence": 1, "jira": 3}
    assert stats["skipped_questions_by_reason"] == {"ambiguous_expected_documents": 2}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
