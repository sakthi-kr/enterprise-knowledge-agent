"""Grounded RAG answer assembly over retrieved enterprise evidence."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from enterprise_knowledge_agent.vector_search import RetrievalHit, VectorRetriever

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


class GroundedLanguageModel(Protocol):
    """Language-model interface required by the RAG service."""

    def generate(
        self,
        *,
        question: str,
        evidence: Sequence[EvidenceSource],
    ) -> LanguageModelOutput:
        """Generate one structured answer from retrieved evidence."""


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

        selected: list[EvidenceSource] = []
        per_document: defaultdict[str, int] = defaultdict(int)
        used_characters = 0

        for hit in hits:
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
                )
            )
            per_document[hit.doc_id] += 1
            used_characters += len(text)

        return selected


class GroundedAnswerService:
    """Retrieve evidence, generate an answer, and derive citations from the answer text."""

    def __init__(
        self,
        *,
        retriever: VectorRetriever,
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
        if not evidence:
            return GroundedAnswer(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=(
                    "I don't have enough evidence in the indexed enterprise data to answer that."
                ),
                citations=(),
                model_name="not_called",
                usage=TokenUsage(),
                retrieved_chunk_count=len(hits),
                context_source_count=0,
            )

        output = self._language_model.generate(question=question, evidence=evidence)
        status, answer, citations = self._validate_output(output, evidence=evidence)
        return GroundedAnswer(
            status=status,
            answer=answer,
            citations=citations,
            model_name=output.model_name,
            usage=output.usage,
            retrieved_chunk_count=len(hits),
            context_source_count=len(evidence),
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
