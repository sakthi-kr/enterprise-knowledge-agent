"""Tests for the EnterpriseRAG-Bench subset downloader."""

import json
import zipfile
from pathlib import Path

from enterprise_knowledge_agent.dataset_download import (
    ARCHIVE_FILENAMES,
    QUESTIONS_FILENAME,
    download_subset,
    validate_questions,
    validate_zip,
)


def _write_source_release(root: Path) -> None:
    root.mkdir()
    for index, filename in enumerate(ARCHIVE_FILENAMES, start=1):
        doc_id = f"dsid_{index:032x}"
        with zipfile.ZipFile(root / filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{doc_id}_document.txt", f"Title {index}\nDocument {index}")

    questions = [
        {
            "question_id": "qst_0001",
            "question_type": "Basic",
            "question": "Example?",
            "expected_doc_ids": ["dsid_00000000000000000000000000000001"],
            "gold_answer": "Example.",
        }
    ]
    with (root / QUESTIONS_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(json.dumps(question))
            handle.write("\n")


def test_download_subset_from_local_release_and_reuses_valid_files(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    _write_source_release(release_dir)
    archives_dir = tmp_path / "raw" / "archives"
    questions_file = tmp_path / "raw" / QUESTIONS_FILENAME
    base_url = release_dir.as_uri() + "/"

    first = download_subset(
        archives_dir=archives_dir,
        questions_file=questions_file,
        base_url=base_url,
        retries=0,
        timeout_seconds=5,
    )
    second = download_subset(
        archives_dir=archives_dir,
        questions_file=questions_file,
        base_url=base_url,
        retries=0,
        timeout_seconds=5,
    )

    assert first["archive_count"] == 6
    assert first["question_count"] == 1
    assert sorted(first["downloaded"]) == sorted((*ARCHIVE_FILENAMES, QUESTIONS_FILENAME))
    assert first["reused"] == []
    assert second["downloaded"] == []
    assert sorted(second["reused"]) == sorted((*ARCHIVE_FILENAMES, QUESTIONS_FILENAME))


def test_download_subset_replaces_invalid_existing_file(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    _write_source_release(release_dir)
    archives_dir = tmp_path / "raw" / "archives"
    archives_dir.mkdir(parents=True)
    broken_archive = archives_dir / ARCHIVE_FILENAMES[0]
    broken_archive.write_text("not a zip", encoding="utf-8")

    result = download_subset(
        archives_dir=archives_dir,
        questions_file=tmp_path / "raw" / QUESTIONS_FILENAME,
        base_url=release_dir.as_uri() + "/",
        retries=0,
        timeout_seconds=5,
    )

    validate_zip(broken_archive)
    assert ARCHIVE_FILENAMES[0] in result["downloaded"]


def test_validators_reject_invalid_files(tmp_path: Path) -> None:
    invalid_zip = tmp_path / "bad.zip"
    invalid_zip.write_text("not a zip", encoding="utf-8")
    invalid_questions = tmp_path / "questions.jsonl"
    invalid_questions.write_text('{"not_question_id": 1}\n', encoding="utf-8")

    for validator, path in [
        (validate_zip, invalid_zip),
        (validate_questions, invalid_questions),
    ]:
        try:
            validator(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected validation failure for {path}")
