"""Graph-augmented grounded answer tests."""

from enterprise_knowledge_agent.graph_retrieval import GraphRetrievalHit, GraphRetrievalTrace
from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    GraphAugmentedAnswerService,
    GraphContextBuilder,
    LanguageModelOutput,
    TokenUsage,
)
from enterprise_knowledge_agent.vector_search import RetrievalHit


def _chunk(*, rank: int, record_id: str, doc_id: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=1.0 - rank * 0.01,
        chunk_id=f"chunk-{record_id}",
        record_id=record_id,
        doc_id=doc_id,
        source_type="jira",
        title=f"Title {doc_id}",
        source_file=f"{doc_id}.txt",
        chunk_index=0,
        text=text,
    )


def _graph_hit(
    *,
    rank: int,
    record_id: str,
    doc_id: str,
    matched_entity_count: int,
    sources: tuple[str, ...] = ("graph",),
) -> GraphRetrievalHit:
    return GraphRetrievalHit(
        rank=rank,
        fused_score=1.0 / rank,
        record_id=record_id,
        doc_id=doc_id,
        source_type="confluence",
        title=f"Graph {doc_id}",
        dense_rank=None,
        graph_rank=rank,
        graph_score=2.0,
        matched_entity_count=matched_entity_count,
        matched_entities=("API Gateway", "Autoscaler"),
        retrieval_sources=sources,
    )


class _FakeDenseRetriever:
    def __init__(
        self,
        *,
        dense_hits: list[RetrievalHit],
        record_hits: list[RetrievalHit],
    ) -> None:
        self.dense_hits = dense_hits
        self.record_hits = record_hits
        self.record_calls: list[list[str]] = []

    def search(self, query: str, *, limit: int = 5) -> list[RetrievalHit]:
        assert query == "What caused the incident?"
        return self.dense_hits[:limit]

    def search_records(
        self,
        query: str,
        *,
        record_ids: list[str],
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        assert query == "What caused the incident?"
        self.record_calls.append(list(record_ids))
        allowed = set(record_ids)
        hits = [hit for hit in self.record_hits if hit.record_id in allowed]
        return hits[:limit]


class _FakeGraphRetriever:
    def __init__(self, hits: list[GraphRetrievalHit]) -> None:
        self.hits = hits

    def search_documents_with_trace(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[list[GraphRetrievalHit], GraphRetrievalTrace]:
        assert query == "What caused the incident?"
        trace = GraphRetrievalTrace(
            dense_candidate_count=30,
            seed_document_count=6,
            seed_entity_count=4,
            neighbor_entity_count=8,
            graph_candidate_count=len(self.hits),
            seed_entities=("API Gateway",),
            neighbor_entities=("Autoscaler",),
        )
        return self.hits[:limit], trace


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.evidence_sources: list[str] = []

    def generate(self, *, question, evidence):
        assert question == "What caused the incident?"
        self.evidence_sources = [source.retrieval_source for source in evidence]
        return LanguageModelOutput(
            status=AnswerStatus.ANSWERED,
            answer="Dense evidence and graph context support the answer [S1] [S5].",
            model_name="test-model",
            usage=TokenUsage(total_tokens=10),
        )


def test_graph_context_builder_preserves_dense_head_and_reserves_graph_slots() -> None:
    builder = GraphContextBuilder(
        max_sources=6,
        dense_sources=4,
        graph_sources=2,
        max_per_document=1,
        max_context_characters=1000,
    )
    dense_hits = [
        _chunk(rank=index, record_id=f"dense-{index}", doc_id=f"doc-{index}", text="dense")
        for index in range(1, 7)
    ]
    graph_hits = [
        _chunk(rank=1, record_id="graph-1", doc_id="graph-doc-1", text="graph one"),
        _chunk(rank=2, record_id="graph-2", doc_id="graph-doc-2", text="graph two"),
    ]

    evidence = builder.build_augmented(dense_hits=dense_hits, graph_hits=graph_hits)

    assert [source.record_id for source in evidence] == [
        "dense-1",
        "dense-2",
        "dense-3",
        "dense-4",
        "graph-1",
        "graph-2",
    ]
    assert [source.retrieval_source for source in evidence] == [
        "dense",
        "dense",
        "dense",
        "dense",
        "graph",
        "graph",
    ]


def test_graph_augmented_service_uses_only_strong_graph_only_candidates() -> None:
    dense_hits = [
        _chunk(rank=index, record_id=f"dense-{index}", doc_id=f"doc-{index}", text="dense")
        for index in range(1, 7)
    ]
    record_hits = [
        _chunk(rank=1, record_id="graph-strong", doc_id="graph-doc", text="graph evidence")
    ]
    dense = _FakeDenseRetriever(dense_hits=dense_hits, record_hits=record_hits)
    graph = _FakeGraphRetriever(
        [
            _graph_hit(
                rank=1,
                record_id="dense-1",
                doc_id="doc-1",
                matched_entity_count=4,
                sources=("dense", "graph"),
            ),
            _graph_hit(
                rank=2,
                record_id="graph-weak",
                doc_id="weak-doc",
                matched_entity_count=1,
            ),
            _graph_hit(
                rank=3,
                record_id="graph-strong",
                doc_id="graph-doc",
                matched_entity_count=3,
            ),
        ]
    )
    model = _FakeLanguageModel()
    service = GraphAugmentedAnswerService(
        retriever=dense,
        graph_retriever=graph,
        language_model=model,
        context_builder=GraphContextBuilder(),
        retrieval_candidates=6,
        graph_document_candidates=10,
        graph_fetch_candidates=4,
        min_graph_matched_entities=2,
    )

    result = service.answer("What caused the incident?")

    assert dense.record_calls == [["graph-strong"]]
    assert model.evidence_sources == ["dense", "dense", "dense", "dense", "graph", "dense"]
    assert result.status is AnswerStatus.ANSWERED
    assert result.retrieval_strategy == "dense_with_graph_context"
    assert result.graph_context_source_count == 1
    assert result.graph_candidate_count == 1
    assert [source.citation_id for source in result.citations] == ["S1", "S5"]
    assert result.citations[1].retrieval_source == "graph"


def test_graph_augmented_service_falls_back_to_dense_when_graph_has_no_strong_candidates() -> None:
    dense_hits = [
        _chunk(rank=index, record_id=f"dense-{index}", doc_id=f"doc-{index}", text="dense")
        for index in range(1, 7)
    ]
    dense = _FakeDenseRetriever(dense_hits=dense_hits, record_hits=[])
    graph = _FakeGraphRetriever(
        [
            _graph_hit(
                rank=1,
                record_id="graph-weak",
                doc_id="weak-doc",
                matched_entity_count=1,
            )
        ]
    )
    model = _FakeLanguageModel()
    service = GraphAugmentedAnswerService(
        retriever=dense,
        graph_retriever=graph,
        language_model=model,
        context_builder=GraphContextBuilder(),
        retrieval_candidates=6,
        min_graph_matched_entities=2,
    )

    result = service.answer("What caused the incident?")

    assert dense.record_calls == [[]]
    assert result.graph_context_source_count == 0
    assert result.graph_candidate_count == 0
    assert result.context_source_count == 6
