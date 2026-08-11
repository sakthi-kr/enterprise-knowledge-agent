"""Evaluate graph-assisted retrieval against the dense-vector baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.data_pipeline import read_jsonl
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder
from enterprise_knowledge_agent.graph_retrieval import GraphRAGRetriever
from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore
from enterprise_knowledge_agent.qdrant_store import QdrantStore
from enterprise_knowledge_agent.retrieval_evaluation import evaluate_retrieval
from enterprise_knowledge_agent.vector_search import VectorRetriever

DEFAULT_QUESTIONS = Path("data/processed/enterprise_rag_bench/benchmark_questions.jsonl")
DEFAULT_BASELINE_METRICS = Path("artifacts/retrieval/vector_baseline_metrics.json")
DEFAULT_METRICS = Path("artifacts/retrieval/graphrag_metrics.json")
DEFAULT_RESULTS = Path("artifacts/retrieval/graphrag_results.jsonl")
DEFAULT_COMPARISON = Path("artifacts/retrieval/graphrag_comparison.json")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required metrics file does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _metric_delta(new_value: Any, baseline_value: Any) -> float:
    return round(float(new_value) - float(baseline_value), 6)


def compare_retrieval_metrics(
    *,
    baseline: dict[str, Any],
    graphrag: dict[str, Any],
) -> dict[str, Any]:
    """Compare overall and per-question-type metrics from two retrieval runs."""

    if baseline.get("evaluated_question_count") != graphrag.get("evaluated_question_count"):
        raise ValueError("Baseline and GraphRAG evaluated different question counts")

    baseline_metrics = baseline.get("metrics")
    graphrag_metrics = graphrag.get("metrics")
    if not isinstance(baseline_metrics, dict) or not isinstance(graphrag_metrics, dict):
        raise ValueError("Retrieval metrics payload is missing its metrics object")

    metric_names = sorted(set(baseline_metrics).intersection(graphrag_metrics))
    overall = {
        metric: {
            "dense": float(baseline_metrics[metric]),
            "graphrag": float(graphrag_metrics[metric]),
            "delta": _metric_delta(graphrag_metrics[metric], baseline_metrics[metric]),
        }
        for metric in metric_names
    }

    baseline_by_type = baseline.get("metrics_by_question_type")
    graphrag_by_type = graphrag.get("metrics_by_question_type")
    if not isinstance(baseline_by_type, dict) or not isinstance(graphrag_by_type, dict):
        raise ValueError("Retrieval metrics payload is missing per-type metrics")

    by_type: dict[str, Any] = {}
    for question_type in sorted(set(baseline_by_type).intersection(graphrag_by_type)):
        baseline_row = baseline_by_type[question_type]
        graphrag_row = graphrag_by_type[question_type]
        if not isinstance(baseline_row, dict) or not isinstance(graphrag_row, dict):
            continue
        metrics: dict[str, Any] = {}
        for metric in ("hit_rate@10", "recall@10", "mrr@10"):
            if metric not in baseline_row or metric not in graphrag_row:
                continue
            metrics[metric] = {
                "dense": float(baseline_row[metric]),
                "graphrag": float(graphrag_row[metric]),
                "delta": _metric_delta(graphrag_row[metric], baseline_row[metric]),
            }
        by_type[question_type] = {
            "question_count": int(graphrag_row.get("question_count", 0)),
            "metrics": metrics,
        }

    baseline_latency = float(baseline["mean_query_latency_ms"])
    graphrag_latency = float(graphrag["mean_query_latency_ms"])
    return {
        "evaluated_question_count": int(graphrag["evaluated_question_count"]),
        "overall": overall,
        "latency_ms": {
            "dense": baseline_latency,
            "graphrag": graphrag_latency,
            "delta": round(graphrag_latency - baseline_latency, 3),
        },
        "by_question_type": by_type,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Run GraphRAG retrieval evaluation and compare it with the dense baseline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args()

    settings = get_settings()
    encoder = FastEmbedTextEncoder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    qdrant = QdrantStore(
        base_url=settings.qdrant_url,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    dense = VectorRetriever(
        store=qdrant,
        encoder=encoder,
        collection_name=settings.qdrant_collection,
    )
    retriever = GraphRAGRetriever(
        dense_retriever=dense,
        graph_store=graph,
        dense_candidates=settings.graphrag_dense_candidates,
        seed_documents=settings.graphrag_seed_documents,
        seed_entities=settings.graphrag_seed_entities,
        neighbor_entities=settings.graphrag_neighbor_entities,
        graph_candidates=settings.graphrag_graph_candidates,
        max_entity_document_count=settings.graphrag_max_entity_document_count,
        min_cooccurrence_documents=settings.graphrag_min_cooccurrence_documents,
        rrf_k=settings.graphrag_rrf_k,
        dense_weight=settings.graphrag_dense_weight,
        graph_weight=settings.graphrag_graph_weight,
    )

    try:
        qdrant.health()
        graph.verify_connectivity()
        graphrag_metrics = evaluate_retrieval(
            questions=read_jsonl(args.questions),
            retriever=retriever,
            metrics_path=args.metrics,
            results_path=args.results,
        )
    finally:
        qdrant.close()
        graph.close()

    baseline_metrics = _read_json(args.baseline_metrics)
    comparison = compare_retrieval_metrics(
        baseline=baseline_metrics,
        graphrag=graphrag_metrics,
    )
    _write_json(args.comparison, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
