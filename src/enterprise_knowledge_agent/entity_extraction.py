"""Extract and normalize enterprise entities from the prepared corpus."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.data_pipeline import read_jsonl

DEFAULT_DOCUMENTS_FILE = Path("data/processed/enterprise_rag_bench/documents.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/enterprise_rag_bench/entities")
DEFAULT_ARTIFACT_FILE = Path("artifacts/nlp/entity_extraction_stats.json")
ENTITY_SCHEMA = {
    "person": "Specific named people. Exclude job titles, roles, teams, and generic groups.",
    "organization": (
        "Specific named companies, vendors, customers, or legal organizations. "
        "Exclude internal teams and generic departments."
    ),
    "team": (
        "Specific named internal teams, squads, or working groups. "
        "Exclude generic roles and unnamed groups."
    ),
    "project": (
        "Specific named projects, programs, initiatives, or launches. "
        "Exclude generic work activities and planning terms."
    ),
    "service": (
        "Specific named deployed services, applications, APIs, platforms, or system components. "
        "Exclude generic technical concepts, symptoms, and failure descriptions."
    ),
    "technology": (
        "Specific named technologies, frameworks, databases, protocols, cloud products, tools, "
        "or software products. Exclude generic technical phrases."
    ),
    "repository": (
        "Specific named source-code repositories or repository identifiers. "
        "Exclude generic mentions of code or repositories."
    ),
}
ENTITY_TYPES = tuple(ENTITY_SCHEMA)
SEPARATOR_PATTERN = re.compile(r"[\s_-]+")
TRIM_CHARACTERS = " \t\r\n.,;:!?()[]{}<>\"'`|/\\"


class EntityExtractionBackend(Protocol):
    """Minimal interface used by the corpus extraction pipeline."""

    model_name: str

    def extract_batch(
        self,
        texts: Sequence[str],
        entity_schema: Mapping[str, str],
        *,
        batch_size: int,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        """Extract entity candidates from a batch of texts."""


class GLiNER2Backend:
    """Local GLiNER2 adapter that keeps the heavy dependency optional."""

    def __init__(self, model_name: str) -> None:
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:
            raise RuntimeError(
                "Local entity extraction requires the NLP dependencies. "
                'Install them with: python -m pip install -e ".[dev,nlp]"'
            ) from exc

        self.model_name = model_name
        self._model = GLiNER2.from_pretrained(model_name, map_location="cpu")

    def extract_batch(
        self,
        texts: Sequence[str],
        entity_schema: Mapping[str, str],
        *,
        batch_size: int,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        results = self._model.batch_extract_entities(
            list(texts),
            dict(entity_schema),
            include_confidence=True,
            include_spans=True,
            batch_size=batch_size,
            threshold=confidence_threshold,
        )
        return list(results)


def normalize_entity_key(text: str) -> str:
    """Return a conservative canonical key for matching entity aliases."""

    normalized = unicodedata.normalize("NFKC", text).strip(TRIM_CHARACTERS)
    normalized = SEPARATOR_PATTERN.sub(" ", normalized)
    return normalized.casefold().strip()


def entity_id(entity_type: str, normalized_key: str) -> str:
    """Create a stable UUID for a canonical entity."""

    identity = f"enterprise-knowledge-entity|{entity_type.casefold()}|{normalized_key}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def truncate_for_extraction(text: str, max_characters: int) -> tuple[str, bool]:
    """Bound model input length while preferring a whitespace boundary."""

    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")
    if len(text) <= max_characters:
        return text, False

    candidate = text[:max_characters]
    boundary = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
    if boundary >= max_characters // 2:
        candidate = candidate[:boundary]
    return candidate.rstrip(), True


def _parse_candidate(candidate: Any, source_text: str) -> tuple[str, float, int, int] | None:
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return None
        start = source_text.find(candidate)
        end = start + len(candidate) if start >= 0 else -1
        return text, 1.0, start, end

    if not isinstance(candidate, Mapping):
        return None

    raw_text = candidate.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    try:
        confidence = float(candidate.get("confidence", 1.0))
    except (TypeError, ValueError):
        return None

    start_value = candidate.get("start", -1)
    end_value = candidate.get("end", -1)
    if not isinstance(start_value, int) or not isinstance(end_value, int):
        return None

    text = raw_text.strip()
    start = start_value
    end = end_value
    if start >= 0 and end >= start:
        observed = source_text[start:end]
        if observed.strip() != text:
            start = source_text.find(raw_text)
            end = start + len(raw_text) if start >= 0 else -1
    return text, confidence, start, end


def normalize_extraction_result(
    result: Mapping[str, Any],
    *,
    source_text: str,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Convert one GLiNER2 result into deterministic entity mention records."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")

    raw_entities = result.get("entities", {})
    if not isinstance(raw_entities, Mapping):
        raise ValueError("Entity extraction result must contain an 'entities' mapping")

    mentions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()

    for raw_type, candidates in raw_entities.items():
        entity_type = str(raw_type).strip().casefold()
        if (
            entity_type not in ENTITY_TYPES
            or not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
        ):
            continue

        for candidate in candidates:
            parsed = _parse_candidate(candidate, source_text)
            if parsed is None:
                continue
            text, confidence, start, end = parsed
            if confidence < confidence_threshold:
                continue

            normalized_key = normalize_entity_key(text)
            if len(normalized_key) < 2:
                continue

            identity = (entity_type, normalized_key, start, end)
            if identity in seen:
                continue
            seen.add(identity)

            mentions.append(
                {
                    "entity_id": entity_id(entity_type, normalized_key),
                    "entity_type": entity_type,
                    "text": text,
                    "normalized_key": normalized_key,
                    "confidence": round(confidence, 6),
                    "start": start,
                    "end": end,
                }
            )

    mentions.sort(
        key=lambda item: (
            item["start"] if item["start"] >= 0 else 10**12,
            item["entity_type"],
            item["normalized_key"],
        )
    )
    return mentions


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _load_processed_ids(partial_path: Path) -> set[str]:
    if not partial_path.exists():
        return set()
    return {
        str(record["record_id"])
        for record in read_jsonl(partial_path)
        if isinstance(record.get("record_id"), str)
    }


def _select_display_name(alias_counts: Counter[str]) -> str:
    ranked = sorted(alias_counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))
    return ranked[0][0]


def aggregate_entities(mentions_file: Path, entities_file: Path) -> dict[str, Any]:
    """Aggregate mention records into canonical entity records."""

    aggregates: dict[str, dict[str, Any]] = {}

    for document in read_jsonl(mentions_file):
        record_id = str(document["record_id"])
        source_type = str(document["source_type"])
        mentions = document.get("entities", [])
        if not isinstance(mentions, list):
            raise ValueError(f"Invalid entities list for document {record_id}")

        for mention in mentions:
            if not isinstance(mention, Mapping):
                continue
            canonical_id = str(mention["entity_id"])
            aggregate = aggregates.setdefault(
                canonical_id,
                {
                    "entity_id": canonical_id,
                    "entity_type": str(mention["entity_type"]),
                    "normalized_key": str(mention["normalized_key"]),
                    "aliases": Counter(),
                    "record_ids": set(),
                    "source_types": set(),
                    "mention_count": 0,
                    "max_confidence": 0.0,
                },
            )
            aggregate["aliases"][str(mention["text"])] += 1
            aggregate["record_ids"].add(record_id)
            aggregate["source_types"].add(source_type)
            aggregate["mention_count"] += 1
            aggregate["max_confidence"] = max(
                aggregate["max_confidence"], float(mention["confidence"])
            )

    rows: list[dict[str, Any]] = []
    canonical_by_type: Counter[str] = Counter()
    for aggregate in aggregates.values():
        alias_counts: Counter[str] = aggregate["aliases"]
        entity_type = aggregate["entity_type"]
        canonical_by_type[entity_type] += 1
        rows.append(
            {
                "entity_id": aggregate["entity_id"],
                "entity_type": entity_type,
                "normalized_key": aggregate["normalized_key"],
                "display_name": _select_display_name(alias_counts),
                "aliases": sorted(alias_counts),
                "mention_count": aggregate["mention_count"],
                "document_count": len(aggregate["record_ids"]),
                "source_types": sorted(aggregate["source_types"]),
                "max_confidence": round(aggregate["max_confidence"], 6),
            }
        )

    rows.sort(key=lambda row: (row["entity_type"], row["normalized_key"], row["entity_id"]))
    _write_jsonl(entities_file, rows)
    return {
        "canonical_entity_count": len(rows),
        "canonical_entities_by_type": dict(sorted(canonical_by_type.items())),
    }


def select_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
    sample_per_source: int | None = None,
) -> list[Mapping[str, Any]]:
    """Select a deterministic whole-corpus prefix or balanced source sample."""

    if limit is not None and sample_per_source is not None:
        raise ValueError("limit and sample_per_source cannot be used together")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero when provided")
        return list(documents[:limit])
    if sample_per_source is None:
        return list(documents)
    if sample_per_source <= 0:
        raise ValueError("sample_per_source must be greater than zero when provided")

    selected: list[Mapping[str, Any]] = []
    counts: Counter[str] = Counter()
    source_types = {str(document["source_type"]) for document in documents}
    for document in documents:
        source_type = str(document["source_type"])
        if counts[source_type] >= sample_per_source:
            continue
        selected.append(document)
        counts[source_type] += 1
        if source_types and all(counts[source] >= sample_per_source for source in source_types):
            break
    return selected


def _document_record(
    document: Mapping[str, Any],
    *,
    input_text: str,
    truncated: bool,
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "record_id": str(document["record_id"]),
        "doc_id": str(document["doc_id"]),
        "source_type": str(document["source_type"]),
        "title": str(document["title"]),
        "source_archive": str(document["source_archive"]),
        "source_file": str(document["source_file"]),
        "input_characters": len(input_text),
        "truncated": truncated,
        "entities": entities,
    }


def extract_corpus_entities(
    *,
    documents_file: Path,
    output_dir: Path,
    backend: EntityExtractionBackend,
    batch_size: int,
    confidence_threshold: float,
    max_input_characters: int,
    limit: int | None = None,
    sample_per_source: int | None = None,
) -> dict[str, Any]:
    """Extract canonical entities from corpus documents with resumable local output."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero when provided")
    if sample_per_source is not None and sample_per_source <= 0:
        raise ValueError("sample_per_source must be greater than zero when provided")
    if limit is not None and sample_per_source is not None:
        raise ValueError("limit and sample_per_source cannot be used together")
    if not documents_file.is_file():
        raise FileNotFoundError(f"Prepared documents file not found: {documents_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mentions_file = output_dir / "entity_mentions.jsonl"
    partial_file = output_dir / "entity_mentions.partial.jsonl"
    entities_file = output_dir / "entities.jsonl"
    stats_file = output_dir / "entity_extraction_stats.json"

    if mentions_file.exists():
        raise FileExistsError(
            f"Completed entity output already exists: {mentions_file}. "
            "Use a different output directory for another run."
        )

    processed_ids = _load_processed_ids(partial_file)
    all_documents = list(read_jsonl(documents_file))
    documents = select_documents(
        all_documents,
        limit=limit,
        sample_per_source=sample_per_source,
    )
    documents = [doc for doc in documents if str(doc["record_id"]) not in processed_ids]

    started = time.perf_counter()
    source_counts: Counter[str] = Counter()
    mention_counts: Counter[str] = Counter()
    documents_with_entities = 0
    truncated_documents = 0
    processed_document_count = len(processed_ids)

    if partial_file.exists():
        for existing in read_jsonl(partial_file):
            source_counts[str(existing["source_type"])] += 1
            if existing.get("truncated") is True:
                truncated_documents += 1
            entities = existing.get("entities", [])
            if isinstance(entities, list) and entities:
                documents_with_entities += 1
                for mention in entities:
                    if isinstance(mention, Mapping):
                        mention_counts[str(mention["entity_type"])] += 1

    with partial_file.open("a", encoding="utf-8", newline="\n") as handle:
        for batch_start in range(0, len(documents), batch_size):
            batch_documents = documents[batch_start : batch_start + batch_size]
            texts: list[str] = []
            truncation_flags: list[bool] = []

            for document in batch_documents:
                content = str(document["content"])
                input_text, truncated = truncate_for_extraction(content, max_input_characters)
                texts.append(input_text)
                truncation_flags.append(truncated)

            results = backend.extract_batch(
                texts,
                ENTITY_SCHEMA,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
            )
            if len(results) != len(batch_documents):
                raise RuntimeError(
                    "Entity backend returned a different number of results than input documents"
                )

            for document, input_text, truncated, result in zip(
                batch_documents, texts, truncation_flags, results, strict=True
            ):
                entities = normalize_extraction_result(
                    result,
                    source_text=input_text,
                    confidence_threshold=confidence_threshold,
                )
                record = _document_record(
                    document,
                    input_text=input_text,
                    truncated=truncated,
                    entities=entities,
                )
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()

                processed_document_count += 1
                source_counts[record["source_type"]] += 1
                if truncated:
                    truncated_documents += 1
                if entities:
                    documents_with_entities += 1
                for mention in entities:
                    mention_counts[mention["entity_type"]] += 1

            print(f"Processed {processed_document_count} documents...", flush=True)

    partial_file.replace(mentions_file)
    aggregate_stats = aggregate_entities(mentions_file, entities_file)
    elapsed_seconds = round(time.perf_counter() - started, 3)

    stats = {
        "model_name": backend.model_name,
        "entity_types": list(ENTITY_TYPES),
        "entity_descriptions": dict(ENTITY_SCHEMA),
        "confidence_threshold": confidence_threshold,
        "max_input_characters": max_input_characters,
        "batch_size": batch_size,
        "selection_limit": limit,
        "sample_per_source": sample_per_source,
        "processed_document_count": processed_document_count,
        "documents_with_entities": documents_with_entities,
        "truncated_document_count": truncated_documents,
        "entity_mention_count": sum(mention_counts.values()),
        "entity_mentions_by_type": dict(sorted(mention_counts.items())),
        "source_document_counts": dict(sorted(source_counts.items())),
        "elapsed_seconds": elapsed_seconds,
        **aggregate_stats,
    }
    stats_file.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-file", type=Path, default=DEFAULT_DOCUMENTS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--sample-per-source", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = get_settings()
    backend = GLiNER2Backend(settings.nlp_model)
    stats = extract_corpus_entities(
        documents_file=args.documents_file,
        output_dir=args.output_dir,
        backend=backend,
        batch_size=settings.nlp_batch_size,
        confidence_threshold=settings.nlp_confidence_threshold,
        max_input_characters=settings.nlp_max_input_characters,
        limit=args.limit,
        sample_per_source=args.sample_per_source,
    )

    if (
        args.output_dir == DEFAULT_OUTPUT_DIR
        and args.limit is None
        and args.sample_per_source is None
    ):
        DEFAULT_ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_ARTIFACT_FILE.write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
