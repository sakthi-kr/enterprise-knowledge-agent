"""GraphRAG benchmark comparison tests."""

from __future__ import annotations

import pytest

from enterprise_knowledge_agent.graphrag_evaluation import compare_retrieval_metrics


def _metrics(*, recall: float, latency: float, count: int = 2) -> dict[str, object]:
    return {
        "evaluated_question_count": count,
        "mean_query_latency_ms": latency,
        "metrics": {
            "hit_rate@10": recall,
            "recall@10": recall,
            "mrr@10": recall / 2,
        },
        "metrics_by_question_type": {
            "semantic": {
                "question_count": count,
                "hit_rate@10": recall,
                "recall@10": recall,
                "mrr@10": recall / 2,
            }
        },
    }


def test_compare_retrieval_metrics_reports_quality_and_latency_deltas() -> None:
    comparison = compare_retrieval_metrics(
        baseline=_metrics(recall=0.4, latency=50.0),
        graphrag=_metrics(recall=0.5, latency=80.0),
    )

    assert comparison["overall"]["recall@10"] == {
        "dense": 0.4,
        "graphrag": 0.5,
        "delta": 0.1,
    }
    assert comparison["latency_ms"]["delta"] == 30.0
    assert comparison["by_question_type"]["semantic"]["metrics"]["mrr@10"]["delta"] == 0.05


def test_compare_retrieval_metrics_rejects_mismatched_question_counts() -> None:
    with pytest.raises(ValueError, match="different question counts"):
        compare_retrieval_metrics(
            baseline=_metrics(recall=0.4, latency=50.0, count=2),
            graphrag=_metrics(recall=0.5, latency=80.0, count=3),
        )
