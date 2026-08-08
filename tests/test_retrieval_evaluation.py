"""Tests for retrieval benchmark metrics."""

import json
from pathlib import Path

from enterprise_knowledge_agent.retrieval_evaluation import (
    evaluate_retrieval,
    metrics_for_ranking,
)
from enterprise_knowledge_agent.vector_search import RetrievalHit


def _hit(rank: int, doc_id: str) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        score=1.0 / rank,
        chunk_id=f"chunk-{rank}",
        record_id=f"record-{rank}",
        doc_id=doc_id,
        source_type="jira",
        title=f"Title {rank}",
        source_file=f"jira/{rank}.txt",
        chunk_index=rank - 1,
        text=f"Text {rank}",
    )


class _FakeRetriever:
    def search_documents(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        assert limit == 3
        if query == "alpha":
            return [_hit(1, "doc-x"), _hit(2, "doc-a"), _hit(3, "doc-b")]
        return [_hit(1, "doc-z")]


def test_metrics_for_ranking_uses_document_level_ranks() -> None:
    metrics = metrics_for_ranking(
        ranking=["doc-x", "doc-a", "doc-b"],
        expected_doc_ids={"doc-a", "doc-b"},
        k_values=(1, 3),
    )

    assert metrics == {
        "hit_rate@1": 0.0,
        "recall@1": 0.0,
        "mrr@1": 0.0,
        "hit_rate@3": 1.0,
        "recall@3": 1.0,
        "mrr@3": 0.5,
    }


def test_evaluate_retrieval_writes_metrics_and_results(tmp_path: Path) -> None:
    questions = [
        {
            "question_id": "q1",
            "question_type": "Basic",
            "question": "alpha",
            "expected_doc_ids": ["doc-a", "doc-b"],
        },
        {
            "question_id": "q2",
            "question_type": "Semantic",
            "question": "beta",
            "expected_doc_ids": ["doc-y"],
        },
        {
            "question_id": "q3",
            "question_type": "Info Not Found",
            "question": "missing",
            "expected_doc_ids": [],
        },
    ]
    metrics_path = tmp_path / "metrics.json"
    results_path = tmp_path / "results.jsonl"

    metrics = evaluate_retrieval(
        questions=questions,
        retriever=_FakeRetriever(),
        metrics_path=metrics_path,
        results_path=results_path,
        k_values=(1, 3),
    )

    assert metrics["evaluated_question_count"] == 2
    assert metrics["skipped_questions_without_gold_documents"] == 1
    assert metrics["metrics"]["hit_rate@3"] == 0.5
    assert metrics["metrics"]["recall@3"] == 0.5
    assert metrics["metrics"]["mrr@3"] == 0.25
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 2
