"""Tests for grounded answer assembly and citation validation."""

from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    ContextBuilder,
    EvidenceSource,
    GroundedAnswerService,
    LanguageModelOutput,
    TokenUsage,
)
from enterprise_knowledge_agent.vector_search import RetrievalHit


def _hit(*, rank: int, doc_id: str, text: str, score: float = 0.8) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=score,
        chunk_id=f"chunk-{rank}",
        record_id=f"record-{rank}",
        doc_id=doc_id,
        source_type="jira",
        title=f"Title {rank}",
        source_file=f"source-{rank}.txt",
        chunk_index=rank - 1,
        text=text,
    )


class _FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalHit]:
        self.calls.append((query, limit))
        return self.hits[:limit]


class _FakeLanguageModel:
    def __init__(self, output: LanguageModelOutput) -> None:
        self.output = output
        self.calls: list[tuple[str, list[EvidenceSource]]] = []

    def generate(self, *, question: str, evidence: list[EvidenceSource]) -> LanguageModelOutput:
        self.calls.append((question, list(evidence)))
        return self.output


def _service(hits: list[RetrievalHit], output: LanguageModelOutput) -> GroundedAnswerService:
    return GroundedAnswerService(
        retriever=_FakeRetriever(hits),  # type: ignore[arg-type]
        language_model=_FakeLanguageModel(output),
        context_builder=ContextBuilder(),
        retrieval_candidates=12,
    )


def test_context_builder_limits_chunks_per_document() -> None:
    builder = ContextBuilder(max_sources=3, max_per_document=1, max_context_characters=1000)
    hits = [
        _hit(rank=1, doc_id="doc-a", text="first"),
        _hit(rank=2, doc_id="doc-a", text="second"),
        _hit(rank=3, doc_id="doc-b", text="third"),
    ]

    evidence = builder.build(hits)

    assert [source.citation_id for source in evidence] == ["S1", "S2"]
    assert [source.doc_id for source in evidence] == ["doc-a", "doc-b"]
    assert [source.text for source in evidence] == ["first", "third"]


def test_context_builder_respects_character_budget() -> None:
    builder = ContextBuilder(max_sources=3, max_per_document=2, max_context_characters=8)

    evidence = builder.build([_hit(rank=1, doc_id="doc-a", text="abcdefghijk")])

    assert len(evidence) == 1
    assert evidence[0].text == "abcdefgh"


def test_grounded_answer_service_derives_citations_from_inline_markers() -> None:
    hits = [
        _hit(rank=1, doc_id="doc-a", text="The incident was caused by a timeout."),
        _hit(rank=2, doc_id="doc-b", text="The fix increased the timeout."),
    ]
    retriever = _FakeRetriever(hits)
    model = _FakeLanguageModel(
        LanguageModelOutput(
            status=AnswerStatus.ANSWERED,
            answer="A timeout caused the incident [S1], and the fix increased it [S2].",
            model_name="test-model",
            usage=TokenUsage(prompt_tokens=100, output_tokens=20, total_tokens=120),
        )
    )
    service = GroundedAnswerService(
        retriever=retriever,  # type: ignore[arg-type]
        language_model=model,
        context_builder=ContextBuilder(),
        retrieval_candidates=12,
    )

    result = service.answer("What caused the incident?")

    assert result.status is AnswerStatus.ANSWERED
    assert [citation.citation_id for citation in result.citations] == ["S1", "S2"]
    assert result.retrieved_chunk_count == 2
    assert result.context_source_count == 2
    assert retriever.calls == [("What caused the incident?", 12)]


def test_grounded_answer_service_uses_only_sources_referenced_inline() -> None:
    hits = [
        _hit(rank=1, doc_id="doc-a", text="The crash loop caused the outage."),
        _hit(rank=2, doc_id="doc-b", text="A second source contains supporting context."),
    ]
    output = LanguageModelOutput(
        status=AnswerStatus.ANSWERED,
        answer="The autoscaler crash loop caused the outage [S1].",
        model_name="test-model",
        usage=TokenUsage(),
    )

    result = _service(hits, output).answer("What caused the outage?")

    assert result.status is AnswerStatus.ANSWERED
    assert [citation.citation_id for citation in result.citations] == ["S1"]


def test_grounded_answer_service_accepts_abstention_without_citations() -> None:
    output = LanguageModelOutput(
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        answer="The available evidence does not establish that.",
        model_name="test-model",
        usage=TokenUsage(),
    )

    result = _service([_hit(rank=1, doc_id="doc-a", text="Unrelated evidence")], output).answer(
        "Who approved it?"
    )

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer == "The available evidence does not establish that."
    assert result.citations == ()


def test_grounded_answer_service_downgrades_unknown_inline_citation() -> None:
    output = LanguageModelOutput(
        status=AnswerStatus.ANSWERED,
        answer="The evidence supports the answer [S9].",
        model_name="test-model",
        usage=TokenUsage(),
    )

    result = _service([_hit(rank=1, doc_id="doc-a", text="Evidence")], output).answer("Question")

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.citations == ()
    assert "fully grounded answer" in result.answer


def test_grounded_answer_service_downgrades_answer_without_inline_citation() -> None:
    output = LanguageModelOutput(
        status=AnswerStatus.ANSWERED,
        answer="The evidence supports the answer.",
        model_name="test-model",
        usage=TokenUsage(),
    )

    result = _service([_hit(rank=1, doc_id="doc-a", text="Evidence")], output).answer("Question")

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.citations == ()
    assert "fully grounded answer" in result.answer


def test_grounded_answer_service_normalizes_cited_abstention() -> None:
    output = LanguageModelOutput(
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        answer="The evidence does not establish the answer [S1].",
        model_name="test-model",
        usage=TokenUsage(),
    )

    result = _service([_hit(rank=1, doc_id="doc-a", text="Evidence")], output).answer("Question")

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.citations == ()
    assert "[S1]" not in result.answer
