"""Evaluate dense-vector retrieval against benchmark document IDs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.data_pipeline import read_jsonl
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder
from enterprise_knowledge_agent.qdrant_store import QdrantStore
from enterprise_knowledge_agent.vector_search import RetrievalHit, VectorRetriever

DEFAULT_K_VALUES = (1, 3, 5, 10)


class Retriever(Protocol):
    """Document search interface required by retrieval evaluation."""

    def search_documents(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        """Retrieve ranked documents represented by their highest-scoring chunk."""


def metrics_for_ranking(
    *,
    ranking: list[str],
    expected_doc_ids: set[str],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Compute hit rate, document recall, and reciprocal rank for one question."""

    if not expected_doc_ids:
        raise ValueError("expected_doc_ids must not be empty")
    metrics: dict[str, float] = {}
    for k in k_values:
        if k <= 0:
            raise ValueError("k values must be greater than zero")
        top_k = ranking[:k]
        matched = expected_doc_ids.intersection(top_k)
        metrics[f"hit_rate@{k}"] = 1.0 if matched else 0.0
        metrics[f"recall@{k}"] = len(matched) / len(expected_doc_ids)
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in expected_doc_ids:
                reciprocal_rank = 1.0 / rank
                break
        metrics[f"mrr@{k}"] = reciprocal_rank
    return metrics


def evaluate_retrieval(
    *,
    questions: Iterable[dict[str, Any]],
    retriever: Retriever,
    metrics_path: Path,
    results_path: Path,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate retriever quality and persist aggregate and per-question results."""

    max_k = max(k_values)
    per_question: list[dict[str, Any]] = []
    skipped_without_gold = 0
    type_counts: Counter[str] = Counter()

    for question in questions:
        expected_raw = question.get("expected_doc_ids")
        if not isinstance(expected_raw, list):
            raise ValueError(f"Question {question.get('question_id')} has invalid expected_doc_ids")
        expected = {str(doc_id).lower() for doc_id in expected_raw}
        if not expected:
            skipped_without_gold += 1
            continue

        question_text = str(question.get("question", "")).strip()
        if not question_text:
            raise ValueError(f"Question {question.get('question_id')} has empty question text")

        started = time.perf_counter()
        hits = retriever.search_documents(question_text, limit=max_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ranking = [hit.doc_id.lower() for hit in hits]
        row_metrics = metrics_for_ranking(
            ranking=ranking,
            expected_doc_ids=expected,
            k_values=k_values,
        )
        question_type = str(question.get("question_type", "unknown"))
        type_counts[question_type] += 1
        per_question.append(
            {
                "question_id": str(question.get("question_id", "")),
                "question_type": question_type,
                "question": question_text,
                "expected_doc_ids": sorted(expected),
                "retrieved_doc_ids": ranking[:max_k],
                "latency_ms": round(elapsed_ms, 3),
                **row_metrics,
            }
        )

    if not per_question:
        raise ValueError("No questions with gold document IDs were available for evaluation")

    aggregate: dict[str, float] = {}
    for metric_name in per_question[0]:
        if metric_name.startswith(("hit_rate@", "recall@", "mrr@")):
            aggregate[metric_name] = round(
                mean(float(row[metric_name]) for row in per_question),
                6,
            )

    by_type: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        grouped[str(row["question_type"])].append(row)
    for question_type, rows in sorted(grouped.items()):
        by_type[question_type] = {
            "question_count": len(rows),
            f"hit_rate@{max_k}": round(
                mean(float(row[f"hit_rate@{max_k}"]) for row in rows),
                6,
            ),
            f"recall@{max_k}": round(
                mean(float(row[f"recall@{max_k}"]) for row in rows),
                6,
            ),
            f"mrr@{max_k}": round(
                mean(float(row[f"mrr@{max_k}"]) for row in rows),
                6,
            ),
        }

    metrics: dict[str, Any] = {
        "evaluated_question_count": len(per_question),
        "skipped_questions_without_gold_documents": skipped_without_gold,
        "question_counts_by_type": dict(sorted(type_counts.items())),
        "mean_query_latency_ms": round(mean(float(row["latency_ms"]) for row in per_question), 3),
        "metrics": aggregate,
        "metrics_by_question_type": by_type,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with results_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_question:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Evaluate dense-vector retrieval.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/processed/enterprise_rag_bench/benchmark_questions.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/retrieval/vector_baseline_metrics.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/retrieval/vector_baseline_results.jsonl"),
    )
    return parser


def main() -> None:
    """Run retrieval evaluation against the configured vector collection."""

    args = build_parser().parse_args()
    settings = get_settings()
    encoder = FastEmbedTextEncoder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    with QdrantStore(
        base_url=settings.qdrant_url,
        timeout_seconds=settings.qdrant_timeout_seconds,
    ) as store:
        store.health()
        retriever = VectorRetriever(
            store=store,
            encoder=encoder,
            collection_name=settings.qdrant_collection,
        )
        metrics = evaluate_retrieval(
            questions=read_jsonl(args.questions),
            retriever=retriever,
            metrics_path=args.metrics,
            results_path=args.results,
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
