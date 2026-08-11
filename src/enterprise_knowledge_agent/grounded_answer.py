"""Grounded answer assembly over dense and graph-augmented enterprise evidence."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from enterprise_knowledge_agent.graph_retrieval import GraphRetrievalHit, GraphRetrievalTrace
from enterprise_knowledge_agent.vector_search import RetrievalHit

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_GROUNDING_FALLBACK = (
    "I couldn't produce a fully grounded answer from the retrieved enterprise evidence."
)


class AnswerStatus(str, Enum):
    """Supported grounded-answer outcomes."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class EvidenceSource:
    """A retrieved chunk exposed to the language model with a stable citation label."""

    citation_id: str
    rank: int
    score: float
    chunk_id: str
    record_id: str
    doc_id: str
    source_type: str
    title: str
    source_file: str
    text: str
    retrieval_source: str = "dense"


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the language-model provider."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LanguageModelOutput:
    """Structured output returned by a grounded language model."""

    status: AnswerStatus
    answer: str
    model_name: str
    usage: TokenUsage


@dataclass(frozen=True)
class GroundedAnswer:
    """Validated application answer with only the sources actually cited."""

    status: AnswerStatus
    answer: str
    citations: tuple[EvidenceSource, ...]
    model_name: str
    usage: TokenUsage
    retrieved_chunk_count: int
    context_source_count: int
    retrieval_strategy: str = "dense"
    graph_context_source_count: int = 0
    graph_candidate_count: int = 0


class GroundedLanguageModel(Protocol):
    """Language-model interface required by the RAG service."""

    def generate(
        self,
        *,
        question: str,
        evidence: Sequence[EvidenceSource],
    ) -> LanguageModelOutput:
        """Generate one structured answer from retrieved evidence."""


class DenseChunkRetriever(Protocol):
    """Dense chunk retrieval required by grounded answer services."""

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalHit]:
        """Return ranked dense chunks."""

    def search_records(
        self,
        query: str,
        *,
        record_ids: list[str],
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        """Return best query-matching chunks from selected physical documents."""


class GraphDocumentRetriever(Protocol):
    """Graph-assisted document retrieval required for context augmentation."""

    def search_documents_with_trace(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[list[GraphRetrievalHit], GraphRetrievalTrace]:
        """Return graph-assisted document ranking and diagnostics."""


class ContextBuilder:
    """Select a small, diverse evidence set from ranked chunk retrieval."""

    def __init__(
        self,
        *,
        max_sources: int = 6,
        max_per_document: int = 2,
        max_context_characters: int = 18000,
    ) -> None:
        if max_sources <= 0:
            raise ValueError("max_sources must be greater than zero")
        if max_per_document <= 0:
            raise ValueError("max_per_document must be greater than zero")
        if max_context_characters <= 0:
            raise ValueError("max_context_characters must be greater than zero")
        self._max_sources = max_sources
        self._max_per_document = max_per_document
        self._max_context_characters = max_context_characters

    def build(self, hits: Sequence[RetrievalHit]) -> list[EvidenceSource]:
        """Select ranked chunks while limiting repeated documents and context size."""

        return self._build_sources([(hit, "dense") for hit in hits])

    def _build_sources(
        self,
        candidates: Sequence[tuple[RetrievalHit, str]],
    ) -> list[EvidenceSource]:
        selected: list[EvidenceSource] = []
        per_document: defaultdict[str, int] = defaultdict(int)
        used_characters = 0

        for hit, retrieval_source in candidates:
            if len(selected) >= self._max_sources:
                break
            if per_document[hit.doc_id] >= self._max_per_document:
                continue

            remaining = self._max_context_characters - used_characters
            if remaining <= 0:
                break
            text = hit.text.strip()
            if not text:
                continue
            if len(text) > remaining:
                text = text[:remaining].rstrip()
            if not text:
                break

            citation_id = f"S{len(selected) + 1}"
            selected.append(
                EvidenceSource(
                    citation_id=citation_id,
                    rank=hit.rank,
                    score=hit.score,
                    chunk_id=hit.chunk_id,
                    record_id=hit.record_id,
                    doc_id=hit.doc_id,
                    source_type=hit.source_type,
                    title=hit.title,
                    source_file=hit.source_file,
                    text=text,
                    retrieval_source=retrieval_source,
                )
            )
            per_document[hit.doc_id] += 1
            used_characters += len(text)

        return selected


class GraphContextBuilder(ContextBuilder):
    """Preserve dense evidence while reserving limited room for graph-only context."""

    def __init__(
        self,
        *,
        max_sources: int = 6,
        dense_sources: int = 4,
        graph_sources: int = 2,
        max_per_document: int = 2,
        max_context_characters: int = 18000,
    ) -> None:
        super().__init__(
            max_sources=max_sources,
            max_per_document=max_per_document,
            max_context_characters=max_context_characters,
        )
        if dense_sources <= 0:
            raise ValueError("dense_sources must be greater than zero")
        if graph_sources <= 0:
            raise ValueError("graph_sources must be greater than zero")
        if dense_sources + graph_sources > max_sources:
            raise ValueError("dense_sources + graph_sources must not exceed max_sources")
        self._dense_sources = dense_sources
        self._graph_sources = graph_sources

    def build_augmented(
        self,
        *,
        dense_hits: Sequence[RetrievalHit],
        graph_hits: Sequence[RetrievalHit],
    ) -> list[EvidenceSource]:
        """Build dense-first evidence with a bounded graph-only supplement."""

        dense_head = list(dense_hits[: self._dense_sources])
        selected_records = {hit.record_id for hit in dense_head}
        graph_head: list[RetrievalHit] = []
        for hit in graph_hits:
            if hit.record_id in selected_records:
                continue
            graph_head.append(hit)
            selected_records.add(hit.record_id)
            if len(graph_head) >= self._graph_sources:
                break

        candidates: list[tuple[RetrievalHit, str]] = [(hit, "dense") for hit in dense_head]
        candidates.extend((hit, "graph") for hit in graph_head)
        candidates.extend(
            (hit, "dense")
            for hit in dense_hits[self._dense_sources :]
            if hit.record_id not in selected_records
        )
        return self._build_sources(candidates)


class GroundedAnswerService:
    """Retrieve dense evidence, generate an answer, and validate inline citations."""

    def __init__(
        self,
        *,
        retriever: DenseChunkRetriever,
        language_model: GroundedLanguageModel,
        context_builder: ContextBuilder,
        retrieval_candidates: int = 12,
    ) -> None:
        if retrieval_candidates <= 0:
            raise ValueError("retrieval_candidates must be greater than zero")
        self._retriever = retriever
        self._language_model = language_model
        self._context_builder = context_builder
        self._retrieval_candidates = retrieval_candidates

    def answer(self, question: str) -> GroundedAnswer:
        """Answer a non-empty question using only retrieved enterprise evidence."""

        if not question.strip():
            raise ValueError("question must not be empty")

        hits = self._retriever.search(question, limit=self._retrieval_candidates)
        evidence = self._context_builder.build(hits)
        return self._generate_answer(
            question=question,
            evidence=evidence,
            retrieved_chunk_count=len(hits),
        )

    def _generate_answer(
        self,
        *,
        question: str,
        evidence: Sequence[EvidenceSource],
        retrieved_chunk_count: int,
        retrieval_strategy: str = "dense",
        graph_candidate_count: int = 0,
    ) -> GroundedAnswer:
        if not evidence:
            return GroundedAnswer(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=(
                    "I don't have enough evidence in the indexed enterprise data to answer that."
                ),
                citations=(),
                model_name="not_called",
                usage=TokenUsage(),
                retrieved_chunk_count=retrieved_chunk_count,
                context_source_count=0,
                retrieval_strategy=retrieval_strategy,
                graph_context_source_count=0,
                graph_candidate_count=graph_candidate_count,
            )

        output = self._language_model.generate(question=question, evidence=evidence)
        status, answer, citations = self._validate_output(output, evidence=evidence)
        graph_context_source_count = sum(source.retrieval_source == "graph" for source in evidence)
        return GroundedAnswer(
            status=status,
            answer=answer,
            citations=citations,
            model_name=output.model_name,
            usage=output.usage,
            retrieved_chunk_count=retrieved_chunk_count,
            context_source_count=len(evidence),
            retrieval_strategy=retrieval_strategy,
            graph_context_source_count=graph_context_source_count,
            graph_candidate_count=graph_candidate_count,
        )

    @staticmethod
    def _validate_output(
        output: LanguageModelOutput,
        *,
        evidence: Sequence[EvidenceSource],
    ) -> tuple[AnswerStatus, str, tuple[EvidenceSource, ...]]:
        answer = output.answer.strip()
        if not answer:
            return AnswerStatus.INSUFFICIENT_EVIDENCE, _GROUNDING_FALLBACK, ()

        inline_ids = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))

        if output.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
            if inline_ids:
                return AnswerStatus.INSUFFICIENT_EVIDENCE, _GROUNDING_FALLBACK, ()
            return AnswerStatus.INSUFFICIENT_EVIDENCE, answer, ()

        evidence_by_id = {source.citation_id: source for source in evidence}
        if not inline_ids:
            return AnswerStatus.INSUFFICIENT_EVIDENCE, _GROUNDING_FALLBACK, ()

        if any(citation_id not in evidence_by_id for citation_id in inline_ids):
            return AnswerStatus.INSUFFICIENT_EVIDENCE, _GROUNDING_FALLBACK, ()

        citations = tuple(evidence_by_id[citation_id] for citation_id in inline_ids)
        return AnswerStatus.ANSWERED, answer, citations


class GraphAugmentedAnswerService(GroundedAnswerService):
    """Generate grounded answers with dense-first context plus graph-only evidence."""

    def __init__(
        self,
        *,
        retriever: DenseChunkRetriever,
        graph_retriever: GraphDocumentRetriever,
        language_model: GroundedLanguageModel,
        context_builder: GraphContextBuilder,
        retrieval_candidates: int = 12,
        graph_document_candidates: int = 10,
        graph_fetch_candidates: int = 4,
        min_graph_matched_entities: int = 2,
    ) -> None:
        super().__init__(
            retriever=retriever,
            language_model=language_model,
            context_builder=context_builder,
            retrieval_candidates=retrieval_candidates,
        )
        if graph_document_candidates <= 0:
            raise ValueError("graph_document_candidates must be greater than zero")
        if graph_fetch_candidates <= 0:
            raise ValueError("graph_fetch_candidates must be greater than zero")
        if min_graph_matched_entities <= 0:
            raise ValueError("min_graph_matched_entities must be greater than zero")
        self._graph_retriever = graph_retriever
        self._graph_context_builder = context_builder
        self._graph_document_candidates = graph_document_candidates
        self._graph_fetch_candidates = graph_fetch_candidates
        self._min_graph_matched_entities = min_graph_matched_entities

    def answer(self, question: str) -> GroundedAnswer:
        """Answer with dense evidence plus a small graph-derived evidence supplement."""

        if not question.strip():
            raise ValueError("question must not be empty")

        dense_hits = self._retriever.search(question, limit=self._retrieval_candidates)
        graph_docs, _ = self._graph_retriever.search_documents_with_trace(
            question,
            limit=self._graph_document_candidates,
        )
        dense_record_ids = {hit.record_id for hit in dense_hits}
        graph_record_ids: list[str] = []
        for hit in graph_docs:
            if "graph" not in hit.retrieval_sources:
                continue
            if hit.record_id in dense_record_ids:
                continue
            if hit.matched_entity_count < self._min_graph_matched_entities:
                continue
            graph_record_ids.append(hit.record_id)
            if len(graph_record_ids) >= self._graph_fetch_candidates:
                break

        graph_hits = self._retriever.search_records(
            question,
            record_ids=graph_record_ids,
            limit=len(graph_record_ids) if graph_record_ids else None,
        )
        evidence = self._graph_context_builder.build_augmented(
            dense_hits=dense_hits,
            graph_hits=graph_hits,
        )
        return self._generate_answer(
            question=question,
            evidence=evidence,
            retrieved_chunk_count=len(dense_hits) + len(graph_hits),
            retrieval_strategy="dense_with_graph_context",
            graph_candidate_count=len(graph_record_ids),
        )
