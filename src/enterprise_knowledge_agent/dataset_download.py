"""Download the local EnterpriseRAG-Bench working subset."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urljoin

DEFAULT_RELEASE_BASE_URL = (
    "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/"
)
ARCHIVE_FILENAMES = (
    "confluence_slice_0001.zip",
    "confluence_slice_0002.zip",
    "jira_slice_0001.zip",
    "jira_slice_0002.zip",
    "github_slice_0001.zip",
    "github_slice_0002.zip",
)
QUESTIONS_FILENAME = "questions.jsonl"


def validate_zip(path: Path) -> None:
    """Raise ValueError when a downloaded ZIP is missing or corrupt."""

    try:
        with zipfile.ZipFile(path) as archive:
            if not any(
                not member.is_dir() and member.filename.lower().endswith(".txt")
                for member in archive.infolist()
            ):
                raise ValueError(f"ZIP contains no .txt documents: {path}")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Corrupt ZIP member in {path}: {bad_member}")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP file: {path}") from exc


def validate_questions(path: Path) -> int:
    """Validate the benchmark JSONL file and return its question count."""

    question_count = 0
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} on line {line_number}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
                    raise ValueError(f"Invalid question record in {path} on line {line_number}")
                question_count += 1
    except UnicodeDecodeError as exc:
        raise ValueError(f"Questions file is not valid UTF-8: {path}") from exc

    if question_count == 0:
        raise ValueError(f"Questions file contains no questions: {path}")
    return question_count


def _download_once(url: str, destination: Path, timeout_seconds: float) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "enterprise-knowledge-agent/0.1.0"},
    )
    with (
        urllib.request.urlopen(request, timeout=timeout_seconds) as response,
        destination.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)


def download_file(
    *,
    url: str,
    destination: Path,
    retries: int,
    timeout_seconds: float,
) -> None:
    """Download one file atomically, retrying transient URL/network failures."""

    if retries < 0:
        raise ValueError("retries must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    temporary.unlink(missing_ok=True)

    for attempt in range(retries + 1):
        try:
            _download_once(url, temporary, timeout_seconds)
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError):
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 5))


def _is_valid_existing(path: Path, *, is_questions: bool) -> bool:
    if not path.is_file():
        return False
    try:
        if is_questions:
            validate_questions(path)
        else:
            validate_zip(path)
    except ValueError:
        return False
    return True


def download_subset(
    *,
    archives_dir: Path,
    questions_file: Path,
    base_url: str = DEFAULT_RELEASE_BASE_URL,
    retries: int = 3,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Download and validate the recommended local benchmark subset."""

    downloaded: list[str] = []
    reused: list[str] = []

    for filename in ARCHIVE_FILENAMES:
        destination = archives_dir / filename
        if _is_valid_existing(destination, is_questions=False):
            reused.append(filename)
            continue

        destination.unlink(missing_ok=True)
        print(f"Downloading {filename} ...")
        download_file(
            url=urljoin(base_url, filename),
            destination=destination,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        validate_zip(destination)
        downloaded.append(filename)

    if _is_valid_existing(questions_file, is_questions=True):
        reused.append(QUESTIONS_FILENAME)
    else:
        questions_file.unlink(missing_ok=True)
        print(f"Downloading {QUESTIONS_FILENAME} ...")
        download_file(
            url=urljoin(base_url, QUESTIONS_FILENAME),
            destination=questions_file,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        validate_questions(questions_file)
        downloaded.append(QUESTIONS_FILENAME)

    question_count = validate_questions(questions_file)
    result: dict[str, object] = {
        "archive_count": len(ARCHIVE_FILENAMES),
        "downloaded": downloaded,
        "question_count": question_count,
        "reused": reused,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Download the recommended EnterpriseRAG-Bench source subset."
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
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main() -> None:
    """Download the benchmark subset from the command line."""

    args = build_parser().parse_args()
    download_subset(
        archives_dir=args.archives_dir,
        questions_file=args.questions_file,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
