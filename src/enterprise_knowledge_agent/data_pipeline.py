"""Prepare an EnterpriseRAG-Bench subset for retrieval experiments."""

from __future__ import annotations

import argparse
import json
import re
import uuid
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO

ARCHIVE_NAME_PATTERN = re.compile(
    r"^(?P<source>[a-z0-9_]+)_slice_\d{4}\.zip$",
    flags=re.IGNORECASE,
)
DOCUMENT_ID_PATTERN = re.compile(
    r"^(?P<doc_id>dsid_[0-9a-f]{32})(?:[_\-.]|$)",
    flags=re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\S+")
INFO_NOT_FOUND = "info not found"


def normalize_text(text: str) -> str:
    """Normalize line endings and excessive blank lines without flattening paragraphs."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def source_type_from_archive(archive_path: Path) -> str:
    """Extract the source type from an EnterpriseRAG-Bench slice filename."""

    match = ARCHIVE_NAME_PATTERN.fullmatch(archive_path.name)
    if match is None:
        raise ValueError(f"Archive name must match '<source>_slice_NNNN.zip': {archive_path.name}")
    return match.group("source").lower()


def document_id_from_member(member_name: str) -> str:
    """Extract a benchmark document ID from a ZIP member filename."""

    filename = Path(member_name).name
    match = DOCUMENT_ID_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Could not extract document ID from filename: {filename}")
    return match.group("doc_id").lower()


def document_record_id(
    *,
    doc_id: str,
    source_type: str,
    source_archive: str,
    source_file: str,
) -> str:
    """Create a stable unique UUID for one physical benchmark document record."""

    identity = "|".join(
        (
            "enterprise-rag-bench-document",
            doc_id.lower(),
            source_type.lower(),
            source_archive,
            source_file,
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def chunk_record_id(record_id: str, chunk_index: int) -> str:
    """Create a stable UUID for a chunk belonging to a physical document record."""

    identity = f"enterprise-rag-bench-chunk|{record_id}|{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def title_from_content(content: str) -> str:
    """Return the first non-empty line as the exported document title."""

    for line in content.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    raise ValueError("Document content is empty after normalization")


def iter_archive_documents(archive_path: Path) -> Iterator[dict[str, str]]:
    """Yield normalized documents directly from a source ZIP archive."""

    source_type = source_type_from_archive(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            (
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".txt")
            ),
            key=lambda member: member.filename,
        )
        if not members:
            raise ValueError(f"Archive contains no .txt documents: {archive_path}")

        for member in members:
            doc_id = document_id_from_member(member.filename)
            try:
                raw_text = archive.read(member).decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Document is not valid UTF-8: {archive_path.name}:{member.filename}"
                ) from exc

            content = normalize_text(raw_text)
            if not content:
                raise ValueError(f"Document is empty: {archive_path.name}:{member.filename}")

            yield {
                "record_id": document_record_id(
                    doc_id=doc_id,
                    source_type=source_type,
                    source_archive=archive_path.name,
                    source_file=member.filename,
                ),
                "doc_id": doc_id,
                "source_type": source_type,
                "title": title_from_content(content),
                "content": content,
                "source_archive": archive_path.name,
                "source_file": member.filename,
            }


def chunk_document(
    document: Mapping[str, str],
    *,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> Iterator[dict[str, Any]]:
    """Split a document into deterministic overlapping word-window chunks."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_size_words:
        raise ValueError(
            "chunk_overlap_words must be non-negative and smaller than chunk_size_words"
        )

    content = document["content"]
    matches = list(WORD_PATTERN.finditer(content))
    if not matches:
        return

    record_id = document["record_id"]
    step = chunk_size_words - chunk_overlap_words
    start_word = 0
    chunk_index = 0

    while start_word < len(matches):
        end_word = min(start_word + chunk_size_words, len(matches))
        start_char = matches[start_word].start()
        end_char = matches[end_word - 1].end()
        text = content[start_char:end_char]

        yield {
            "chunk_id": chunk_record_id(record_id, chunk_index),
            "record_id": record_id,
            "doc_id": document["doc_id"],
            "source_type": document["source_type"],
            "title": document["title"],
            "source_archive": document["source_archive"],
            "source_file": document["source_file"],
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
            "word_count": end_word - start_word,
            "text": text,
        }

        if end_word == len(matches):
            break
        start_word += step
        chunk_index += 1


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSON objects from a JSONL file with useful line-level errors."""

    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path} on line {line_number}")
            yield record


def _normalized_question_type(question: Mapping[str, Any]) -> str:
    value = question.get("question_type", "")
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _expected_doc_ids(question: Mapping[str, Any]) -> list[str]:
    value = question.get("expected_doc_ids")
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        question_id = question.get("question_id", "<unknown>")
        raise ValueError(f"Question {question_id} has invalid expected_doc_ids")
    return [item.lower() for item in value]


def select_compatible_questions(
    questions: Iterable[dict[str, Any]],
    corpus_doc_ids: set[str],
    ambiguous_doc_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    """Keep questions whose gold evidence is unambiguous in the local subset corpus."""

    selected: list[dict[str, Any]] = []
    selected_by_type: Counter[str] = Counter()
    skipped_by_reason: Counter[str] = Counter()
    ambiguous = {doc_id.lower() for doc_id in (ambiguous_doc_ids or set())}

    for question in questions:
        question_type = _normalized_question_type(question)
        expected_doc_ids = _expected_doc_ids(question)

        if expected_doc_ids:
            if any(doc_id in ambiguous for doc_id in expected_doc_ids):
                skipped_by_reason["ambiguous_expected_documents"] += 1
            elif all(doc_id in corpus_doc_ids for doc_id in expected_doc_ids):
                selected.append(question)
                selected_by_type[str(question.get("question_type", "unknown"))] += 1
            else:
                skipped_by_reason["missing_expected_documents"] += 1
            continue

        if question_type == INFO_NOT_FOUND:
            selected.append(question)
            selected_by_type[str(question.get("question_type", "unknown"))] += 1
        else:
            skipped_by_reason["requires_full_corpus_or_has_no_gold_documents"] += 1

    return selected, selected_by_type, skipped_by_reason


def write_jsonl_record(handle: TextIO, record: Mapping[str, Any]) -> None:
    """Write one deterministic UTF-8 JSONL record."""

    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def prepare_corpus(
    *,
    archives_dir: Path,
    questions_file: Path,
    output_dir: Path,
    chunk_size_words: int = 350,
    chunk_overlap_words: int = 60,
) -> dict[str, Any]:
    """Normalize source archives, create chunks, and select compatible benchmark questions."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_size_words:
        raise ValueError(
            "chunk_overlap_words must be non-negative and smaller than chunk_size_words"
        )
    if not questions_file.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    archives = sorted(archives_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found in: {archives_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    questions_path = output_dir / "benchmark_questions.jsonl"
    stats_path = output_dir / "corpus_stats.json"

    source_counts: Counter[str] = Counter()
    document_id_counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    document_count = 0
    chunk_count = 0
    total_document_words = 0

    documents_tmp = documents_path.with_suffix(".jsonl.tmp")
    chunks_tmp = chunks_path.with_suffix(".jsonl.tmp")
    questions_tmp = questions_path.with_suffix(".jsonl.tmp")
    stats_tmp = stats_path.with_suffix(".json.tmp")

    temp_paths = [documents_tmp, chunks_tmp, questions_tmp, stats_tmp]
    for path in temp_paths:
        path.unlink(missing_ok=True)

    try:
        with (
            documents_tmp.open("w", encoding="utf-8", newline="\n") as documents_handle,
            chunks_tmp.open("w", encoding="utf-8", newline="\n") as chunks_handle,
        ):
            for archive_path in archives:
                for document in iter_archive_documents(archive_path):
                    record_id = document["record_id"]
                    if record_id in record_ids:
                        raise ValueError(f"Duplicate physical document record found: {record_id}")
                    record_ids.add(record_id)

                    doc_id = document["doc_id"]
                    document_id_counts[doc_id] += 1

                    write_jsonl_record(documents_handle, document)
                    document_count += 1
                    source_counts[document["source_type"]] += 1
                    total_document_words += len(WORD_PATTERN.findall(document["content"]))

                    for chunk in chunk_document(
                        document,
                        chunk_size_words=chunk_size_words,
                        chunk_overlap_words=chunk_overlap_words,
                    ):
                        write_jsonl_record(chunks_handle, chunk)
                        chunk_count += 1

        corpus_doc_ids = set(document_id_counts)
        ambiguous_doc_ids = {doc_id for doc_id, count in document_id_counts.items() if count > 1}
        duplicate_document_record_count = sum(
            count - 1 for count in document_id_counts.values() if count > 1
        )

        questions = list(read_jsonl(questions_file))
        selected_questions, selected_by_type, skipped_by_reason = select_compatible_questions(
            questions,
            corpus_doc_ids,
            ambiguous_doc_ids,
        )
        with questions_tmp.open("w", encoding="utf-8", newline="\n") as questions_handle:
            for question in selected_questions:
                write_jsonl_record(questions_handle, question)

        stats: dict[str, Any] = {
            "ambiguous_document_id_count": len(ambiguous_doc_ids),
            "ambiguous_document_ids": sorted(ambiguous_doc_ids),
            "archive_count": len(archives),
            "archives": [archive.name for archive in archives],
            "chunk_count": chunk_count,
            "chunk_overlap_words": chunk_overlap_words,
            "chunk_size_words": chunk_size_words,
            "compatible_question_count": len(selected_questions),
            "compatible_questions_by_type": dict(sorted(selected_by_type.items())),
            "document_count": document_count,
            "duplicate_document_record_count": duplicate_document_record_count,
            "mean_document_words": (
                round(total_document_words / document_count, 2) if document_count else 0.0
            ),
            "question_count_total": len(questions),
            "skipped_questions_by_reason": dict(sorted(skipped_by_reason.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "unique_document_id_count": len(document_id_counts),
        }
        stats_tmp.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        documents_tmp.replace(documents_path)
        chunks_tmp.replace(chunks_path)
        questions_tmp.replace(questions_path)
        stats_tmp.replace(stats_path)
        return stats
    except Exception:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Prepare an EnterpriseRAG-Bench source subset for local retrieval experiments."
    )
    parser.add_argument(
        "--archives-dir",
        type=Path,
        default=Path("data/raw/enterprise_rag_bench/archives"),
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("data/raw/enterprise_rag_bench/questions.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/enterprise_rag_bench"),
    )
    parser.add_argument("--chunk-size-words", type=int, default=350)
    parser.add_argument("--chunk-overlap-words", type=int, default=60)
    return parser


def main() -> None:
    """Run corpus preparation from the command line."""

    args = build_parser().parse_args()
    stats = prepare_corpus(
        archives_dir=args.archives_dir,
        questions_file=args.questions_file,
        output_dir=args.output_dir,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
