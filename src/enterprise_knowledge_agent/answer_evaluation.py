"""Evaluate dense RAG, graph-augmented RAG, and the agent on a balanced benchmark sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from enterprise_knowledge_agent.agent import EnterpriseKnowledgeAgent
from enterprise_knowledge_agent.agent_types import AgentStrategy
from enterprise_knowledge_agent.config import get_settings
from enterprise_knowledge_agent.data_pipeline import read_jsonl
from enterprise_knowledge_agent.embeddings import FastEmbedTextEncoder
from enterprise_knowledge_agent.gemini_client import GeminiRestClient
from enterprise_knowledge_agent.graph_retrieval import GraphRAGRetriever
from enterprise_knowledge_agent.groq_client import GroqRestClient
from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    ContextBuilder,
    GraphAugmentedAnswerService,
    GraphContextBuilder,
    GroundedAnswer,
    GroundedAnswerService,
    TokenUsage,
)
from enterprise_knowledge_agent.language_model_errors import LanguageModelAPIError
from enterprise_knowledge_agent.neo4j_store import Neo4jGraphStore
from enterprise_knowledge_agent.observability import build_tracer
from enterprise_knowledge_agent.qdrant_store import QdrantStore
from enterprise_knowledge_agent.vector_search import VectorRetriever

EVALUATOR_VERSION = "2026-08-multi-provider-local-semantic-v2"
DEFAULT_SYSTEMS = ("dense", "graph", "agent")
SUPPORTED_PROVIDERS = ("gemini", "groq")
_PROVIDER_PRICING_USD_PER_MILLION = {
    ("gemini", "gemini-3.6-flash"): (1.50, 7.50),
    ("groq", "openai/gpt-oss-20b"): (0.075, 0.30),
}
INFO_NOT_FOUND = "info_not_found"
GRAPH_POLICY_TYPES = frozenset({"project_related", "completeness", "conflicting_info"})
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_HTTP_STATUS = re.compile(r"HTTP (?P<status>\d{3})")


class PassageEncoder(Protocol):
    """Minimal embedding interface required by the local semantic proxy scorer."""

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts into a shared vector space."""


class AnswerRunner(Protocol):
    """Runtime system interface used by the evaluation loop."""

    name: str
    provider_name: str
    model_name: str

    def run(self, question: str) -> EvaluationOutput:
        """Execute one system for one benchmark question."""


@dataclass(frozen=True)
class EvaluationOutput:
    """System answer plus optional agent-routing metadata."""

    answer: GroundedAnswer
    planner_strategy: str | None = None
    planner_fallback: bool | None = None
    tool_call_count: int | None = None
    graph_tool_called: bool | None = None
    graph_tool_result_count: int | None = None
    planner_usage: TokenUsage = TokenUsage()


class GroundedAnswerRunner:
    """Adapter for a regular grounded-answer service."""

    def __init__(
        self,
        *,
        name: str,
        service: GroundedAnswerService,
        provider_name: str,
        model_name: str,
    ) -> None:
        self.name = name
        self.provider_name = provider_name
        self.model_name = model_name
        self._service = service

    def run(self, question: str) -> EvaluationOutput:
        return EvaluationOutput(answer=self._service.answer(question))


class AgentAnswerRunner:
    """Adapter exposing agent output in the common evaluation shape."""

    name = "agent"

    def __init__(
        self,
        *,
        agent: EnterpriseKnowledgeAgent,
        provider_name: str,
        model_name: str,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._agent = agent

    def run(self, question: str) -> EvaluationOutput:
        result = self._agent.run(question)
        graph_executions = [item for item in result.tool_trace if item.tool_name == "graph_expand"]
        graph_result_count = sum(item.result_count for item in graph_executions)
        return EvaluationOutput(
            answer=result.answer,
            planner_strategy=result.plan.strategy.value,
            planner_fallback=result.planner_fallback,
            tool_call_count=result.tool_call_count,
            graph_tool_called=bool(graph_executions),
            graph_tool_result_count=graph_result_count,
            planner_usage=result.plan.usage,
        )


def normalize_question_type(value: object) -> str:
    """Normalize benchmark question-type labels for stable comparisons."""

    return "_".join(str(value).strip().lower().replace("-", " ").split())


def select_balanced_questions(
    questions: Iterable[dict[str, Any]],
    *,
    sample_per_type: int,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Select a deterministic, balanced sample without exposing question text in artifacts."""

    if sample_per_type <= 0:
        raise ValueError("sample_per_type must be greater than zero")

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        question_id = str(question.get("question_id", "")).strip()
        question_text = str(question.get("question", "")).strip()
        question_type = normalize_question_type(question.get("question_type", "unknown"))
        if not question_id:
            raise ValueError("benchmark question is missing question_id")
        if not question_text:
            raise ValueError(f"Question {question_id} has empty question text")
        grouped[question_type].append(question)

    selected: list[dict[str, Any]] = []
    for _question_type, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda item: hashlib.sha256(f"{seed}|{item['question_id']}".encode()).hexdigest(),
        )
        selected.extend(ranked[:sample_per_type])
    return selected


def expected_agent_strategy(question: dict[str, Any]) -> AgentStrategy:
    """Return the documented heuristic policy used only to audit planner routing."""

    question_type = normalize_question_type(question.get("question_type", "unknown"))
    expected_raw = question.get("expected_doc_ids")
    expected_count = len(expected_raw) if isinstance(expected_raw, list) else 0
    if question_type in GRAPH_POLICY_TYPES or expected_count > 1:
        return AgentStrategy.DENSE_PLUS_GRAPH
    return AgentStrategy.DENSE_ONLY


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity without introducing an additional numeric dependency."""

    if len(left) != len(right) or not left:
        raise ValueError("vectors must have the same non-zero dimension")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def semantic_answer_proxies(
    *,
    answer: str,
    gold_answer: str,
    answer_facts: Sequence[str],
    encoder: PassageEncoder,
    fact_similarity_threshold: float,
) -> tuple[float, float]:
    """Compute local embedding proxies for reference similarity and fact coverage."""

    if not 0.0 <= fact_similarity_threshold <= 1.0:
        raise ValueError("fact_similarity_threshold must be between zero and one")
    if not answer.strip() or not gold_answer.strip():
        return 0.0, 0.0

    answer_vector, gold_vector = encoder.embed_passages([answer, gold_answer])
    gold_similarity = max(0.0, min(1.0, cosine_similarity(answer_vector, gold_vector)))

    normalized_facts = [fact.strip() for fact in answer_facts if fact.strip()]
    if not normalized_facts:
        return gold_similarity, 1.0

    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT.split(answer) if sentence.strip()]
    if not sentences:
        sentences = [answer.strip()]

    sentence_vectors = encoder.embed_passages(sentences)
    fact_vectors = encoder.embed_passages(normalized_facts)
    covered = 0
    for fact_vector in fact_vectors:
        best = max(cosine_similarity(fact_vector, vector) for vector in sentence_vectors)
        if best >= fact_similarity_threshold:
            covered += 1
    return gold_similarity, covered / len(normalized_facts)


def _usage_total(output: EvaluationOutput) -> TokenUsage:
    answer_usage = output.answer.usage
    planner = output.planner_usage
    return TokenUsage(
        prompt_tokens=answer_usage.prompt_tokens + planner.prompt_tokens,
        output_tokens=answer_usage.output_tokens + planner.output_tokens,
        thinking_tokens=answer_usage.thinking_tokens + planner.thinking_tokens,
        total_tokens=answer_usage.total_tokens + planner.total_tokens,
    )


def estimate_standard_paid_cost_usd(
    usage: TokenUsage,
    *,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Estimate standard paid-tier cost; free-tier executions may still cost zero."""

    if input_price_per_million < 0.0 or output_price_per_million < 0.0:
        raise ValueError("token prices must be non-negative")
    input_cost = usage.prompt_tokens * input_price_per_million / 1_000_000
    billed_output = usage.output_tokens + usage.thinking_tokens
    output_cost = billed_output * output_price_per_million / 1_000_000
    return input_cost + output_cost


def build_result_row(
    *,
    question: dict[str, Any],
    system_name: str,
    output: EvaluationOutput,
    latency_ms: float,
    encoder: PassageEncoder,
    fact_similarity_threshold: float,
    input_price_per_million: float,
    output_price_per_million: float,
    provider_name: str = "gemini",
    model_name: str = "gemini-3.6-flash",
) -> dict[str, Any]:
    """Build one privacy-safe per-system benchmark result row."""

    question_id = str(question.get("question_id", "")).strip()
    question_type = normalize_question_type(question.get("question_type", "unknown"))
    expected_raw = question.get("expected_doc_ids")
    if not isinstance(expected_raw, list) or not all(
        isinstance(doc_id, str) for doc_id in expected_raw
    ):
        raise ValueError(f"Question {question_id} has invalid expected_doc_ids")
    expected_ids = {doc_id.lower() for doc_id in expected_raw}
    cited_ids = {citation.doc_id.lower() for citation in output.answer.citations}
    should_answer = question_type != INFO_NOT_FOUND
    answered = output.answer.status is AnswerStatus.ANSWERED

    if expected_ids:
        matched_ids = expected_ids.intersection(cited_ids)
        document_recall: float | None = len(matched_ids) / len(expected_ids)
        citation_precision: float | None = len(matched_ids) / len(cited_ids) if cited_ids else 0.0
        invalid_extra_docs: int | None = len(cited_ids - expected_ids)
    else:
        document_recall = None
        citation_precision = None
        invalid_extra_docs = None

    if should_answer:
        gold_answer = question.get("gold_answer")
        answer_facts_raw = question.get("answer_facts")
        if not isinstance(gold_answer, str) or not gold_answer.strip():
            raise ValueError(f"Question {question_id} has invalid gold_answer")
        if not isinstance(answer_facts_raw, list) or not all(
            isinstance(fact, str) for fact in answer_facts_raw
        ):
            raise ValueError(f"Question {question_id} has invalid answer_facts")
        if answered:
            gold_similarity, fact_coverage = semantic_answer_proxies(
                answer=output.answer.answer,
                gold_answer=gold_answer,
                answer_facts=answer_facts_raw,
                encoder=encoder,
                fact_similarity_threshold=fact_similarity_threshold,
            )
        else:
            gold_similarity, fact_coverage = 0.0, 0.0
    else:
        gold_similarity, fact_coverage = None, None

    total_usage = _usage_total(output)
    policy_strategy = expected_agent_strategy(question) if system_name == "agent" else None
    planner_alignment = (
        output.planner_strategy == policy_strategy.value
        if policy_strategy is not None and output.planner_strategy is not None
        else None
    )

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "question_id": question_id,
        "question_type": question_type,
        "system": system_name,
        "llm_provider": provider_name,
        "llm_model": model_name,
        "status": output.answer.status.value,
        "answerability_correct": answered == should_answer,
        "expected_document_count": len(expected_ids),
        "cited_document_count": len(cited_ids),
        "document_recall": _round_optional(document_recall),
        "citation_precision": _round_optional(citation_precision),
        "invalid_extra_docs": invalid_extra_docs,
        "gold_answer_similarity_proxy": _round_optional(gold_similarity),
        "answer_fact_coverage_proxy": _round_optional(fact_coverage),
        "latency_ms": round(latency_ms, 3),
        "prompt_tokens": total_usage.prompt_tokens,
        "output_tokens": total_usage.output_tokens,
        "thinking_tokens": total_usage.thinking_tokens,
        "total_tokens": total_usage.total_tokens,
        "estimated_standard_paid_cost_usd": round(
            estimate_standard_paid_cost_usd(
                total_usage,
                input_price_per_million=input_price_per_million,
                output_price_per_million=output_price_per_million,
            ),
            8,
        ),
        "retrieval_strategy": output.answer.retrieval_strategy,
        "graph_context_source_count": output.answer.graph_context_source_count,
        "graph_candidate_count": output.answer.graph_candidate_count,
        "planner_strategy": output.planner_strategy,
        "planner_policy_strategy": policy_strategy.value if policy_strategy is not None else None,
        "planner_policy_alignment": planner_alignment,
        "planner_fallback": output.planner_fallback,
        "tool_call_count": output.tool_call_count,
        "graph_tool_called": output.graph_tool_called,
        "graph_tool_result_count": output.graph_tool_result_count,
    }


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _mean_present(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(mean(values), 6) if values else None


def _latest_rows_by_key(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the newest attempt for each question/system pair."""

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("question_id")), str(row.get("system")))
        latest[key] = row
    return list(latest.values())


def _provider_status_code(exc: BaseException) -> int | None:
    """Extract a provider HTTP status without persisting provider error text."""

    if not isinstance(exc, LanguageModelAPIError):
        return None
    match = _HTTP_STATUS.search(str(exc))
    return int(match.group("status")) if match else None


def _is_transient_provider_error(exc: BaseException) -> bool:
    """Return whether a provider failure is safe to retry after a cooldown."""

    status_code = _provider_status_code(exc)
    return status_code in {408, 422, 429} or (status_code is not None and status_code >= 500)


def summarize_rows(
    *,
    rows: Sequence[dict[str, Any]],
    selected_questions: Sequence[dict[str, Any]],
    systems: Sequence[str],
    sample_per_type: int,
    seed: int,
    fact_similarity_threshold: float,
    input_price_per_million: float,
    output_price_per_million: float,
    provider_name: str | None = None,
    model_name: str | None = None,
    context_sources: int | None = None,
    dense_context_sources: int | None = None,
    graph_context_sources: int | None = None,
    max_context_characters: int | None = None,
) -> dict[str, Any]:
    """Aggregate privacy-safe evaluation results overall and by benchmark category."""

    selected_ids = {str(item["question_id"]) for item in selected_questions}
    matching_rows = [
        row
        for row in rows
        if row.get("evaluator_version") == EVALUATOR_VERSION
        and row.get("question_id") in selected_ids
        and row.get("system") in systems
        and (provider_name is None or row.get("llm_provider") == provider_name)
        and (model_name is None or row.get("llm_model") == model_name)
    ]
    current_rows = _latest_rows_by_key(matching_rows)
    question_counts = Counter(
        normalize_question_type(question.get("question_type", "unknown"))
        for question in selected_questions
    )

    summary_by_system: dict[str, Any] = {}
    grouped_by_system: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current_rows:
        grouped_by_system[str(row["system"])].append(row)

    for system_name in systems:
        system_rows = grouped_by_system.get(system_name, [])
        successful = [row for row in system_rows if row.get("error_type") is None]
        aggregate = _aggregate_system_rows(successful)
        errors = [row for row in system_rows if row.get("error_type") is not None]
        aggregate["completed_question_count"] = len(system_rows)
        aggregate["successful_question_count"] = len(successful)
        aggregate["error_count"] = len(errors)
        aggregate["error_status_counts"] = dict(
            sorted(
                Counter(str(row.get("provider_status_code") or "unknown") for row in errors).items()
            )
        )
        aggregate["expected_question_count"] = len(selected_questions)

        grouped_by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in successful:
            grouped_by_type[str(row["question_type"])].append(row)
        aggregate["by_question_type"] = {
            question_type: {
                "question_count": len(type_rows),
                "answerability_accuracy": _mean_present(type_rows, "answerability_correct"),
                "mean_document_recall": _mean_present(type_rows, "document_recall"),
                "mean_gold_answer_similarity_proxy": _mean_present(
                    type_rows, "gold_answer_similarity_proxy"
                ),
                "mean_answer_fact_coverage_proxy": _mean_present(
                    type_rows, "answer_fact_coverage_proxy"
                ),
                "mean_latency_ms": _mean_present(type_rows, "latency_ms"),
            }
            for question_type, type_rows in sorted(grouped_by_type.items())
        }
        summary_by_system[system_name] = aggregate

    evaluation_complete = all(
        summary_by_system[name].get("successful_question_count") == len(selected_questions)
        and summary_by_system[name].get("error_count") == 0
        for name in systems
    )
    comparisons = (
        {
            system_name: _comparison_against_dense(
                dense=summary_by_system.get("dense", {}),
                candidate=summary_by_system.get(system_name, {}),
            )
            for system_name in systems
            if system_name != "dense" and "dense" in summary_by_system
        }
        if evaluation_complete
        else {}
    )

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_scope": "balanced_local_proxy",
        "leaderboard_comparable": False,
        "evaluation_complete": evaluation_complete,
        "sample_per_type": sample_per_type,
        "sample_seed": seed,
        "selected_question_count": len(selected_questions),
        "question_counts_by_type": dict(sorted(question_counts.items())),
        "systems": list(systems),
        "evaluation_provider": provider_name,
        "evaluation_model": model_name,
        "context_budget": {
            "max_sources": context_sources,
            "dense_sources": dense_context_sources,
            "graph_sources": graph_context_sources,
            "max_characters": max_context_characters,
        },
        "semantic_proxy": {
            "embedding_model": get_settings().embedding_model,
            "fact_similarity_threshold": fact_similarity_threshold,
        },
        "cost_assumptions": {
            "currency": "USD",
            "provider": provider_name,
            "model": model_name,
            "input_price_per_million_tokens": input_price_per_million,
            "output_price_per_million_tokens_including_thinking": output_price_per_million,
            "metric_name": "estimated_standard_paid_cost_usd",
            "free_tier_may_cost_zero": True,
        },
        "metrics_by_system": summary_by_system,
        "comparison_vs_dense": comparisons,
    }


def _aggregate_system_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len(rows),
        "answerability_accuracy": _mean_present(rows, "answerability_correct"),
        "mean_document_recall": _mean_present(rows, "document_recall"),
        "mean_citation_precision": _mean_present(rows, "citation_precision"),
        "mean_invalid_extra_docs": _mean_present(rows, "invalid_extra_docs"),
        "mean_gold_answer_similarity_proxy": _mean_present(rows, "gold_answer_similarity_proxy"),
        "mean_answer_fact_coverage_proxy": _mean_present(rows, "answer_fact_coverage_proxy"),
        "mean_latency_ms": _mean_present(rows, "latency_ms"),
        "mean_total_tokens": _mean_present(rows, "total_tokens"),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
        "estimated_standard_paid_cost_usd": round(
            sum(float(row.get("estimated_standard_paid_cost_usd", 0.0)) for row in rows),
            6,
        ),
    }
    if rows and any(row.get("planner_strategy") is not None for row in rows):
        graph_calls = [row for row in rows if row.get("graph_tool_called") is True]
        summary.update(
            {
                "planner_policy_alignment": _mean_present(rows, "planner_policy_alignment"),
                "planner_fallback_rate": _mean_present(rows, "planner_fallback"),
                "mean_tool_call_count": _mean_present(rows, "tool_call_count"),
                "graph_tool_call_rate": len(graph_calls) / len(rows),
                "graph_tool_yield_rate": (
                    sum(int(row.get("graph_tool_result_count") or 0) > 0 for row in graph_calls)
                    / len(graph_calls)
                    if graph_calls
                    else 0.0
                ),
            }
        )
    return summary


def _comparison_against_dense(
    *,
    dense: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "answerability_accuracy",
        "mean_document_recall",
        "mean_citation_precision",
        "mean_gold_answer_similarity_proxy",
        "mean_answer_fact_coverage_proxy",
        "mean_latency_ms",
        "mean_total_tokens",
        "estimated_standard_paid_cost_usd",
    )
    comparison: dict[str, Any] = {}
    for field in fields:
        dense_value = dense.get(field)
        candidate_value = candidate.get(field)
        if isinstance(dense_value, (int, float)) and isinstance(candidate_value, (int, float)):
            comparison[field] = {
                "dense": dense_value,
                "candidate": candidate_value,
                "delta": round(float(candidate_value) - float(dense_value), 6),
            }
    return comparison


def read_existing_results(path: Path) -> list[dict[str, Any]]:
    """Read resumable JSONL results, returning an empty list when none exist."""

    if not path.exists():
        return []
    return list(read_jsonl(path))


def append_result(path: Path, row: dict[str, Any]) -> None:
    """Persist one result immediately so interrupted evaluations can resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def evaluate_systems(
    *,
    questions: Sequence[dict[str, Any]],
    runners: Sequence[AnswerRunner],
    encoder: PassageEncoder,
    results_path: Path,
    fact_similarity_threshold: float,
    input_price_per_million: float,
    output_price_per_million: float,
    pause_seconds: float = 8.0,
    transient_cooldown_seconds: float = 45.0,
    max_pair_attempts: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Run missing/failed pairs with pacing and resumable transient retries."""

    if pause_seconds < 0.0:
        raise ValueError("pause_seconds must not be negative")
    if transient_cooldown_seconds < 0.0:
        raise ValueError("transient_cooldown_seconds must not be negative")
    if max_pair_attempts <= 0:
        raise ValueError("max_pair_attempts must be greater than zero")

    if not runners:
        raise ValueError("at least one answer runner is required")
    provider_models = {(runner.provider_name, runner.model_name) for runner in runners}
    if len(provider_models) != 1:
        raise ValueError("all evaluation runners must use the same provider and model")
    provider_name, model_name = next(iter(provider_models))

    existing = read_existing_results(results_path)
    matching_existing = [
        row
        for row in existing
        if row.get("evaluator_version") == EVALUATOR_VERSION
        and row.get("llm_provider") == provider_name
        and row.get("llm_model") == model_name
    ]
    latest_existing = {
        (str(row.get("question_id")), str(row.get("system"))): row
        for row in _latest_rows_by_key(matching_existing)
    }
    completed = {key for key, row in latest_existing.items() if row.get("error_type") is None}

    executed_any = False
    for question in questions:
        question_id = str(question["question_id"])
        question_text = str(question["question"])
        for runner in runners:
            key = (question_id, runner.name)
            if key in completed:
                continue

            for pair_attempt in range(1, max_pair_attempts + 1):
                if executed_any and pause_seconds > 0.0:
                    sleep(pause_seconds)
                executed_any = True
                started = time.perf_counter()
                try:
                    output = runner.run(question_text)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    row = build_result_row(
                        question=question,
                        system_name=runner.name,
                        output=output,
                        latency_ms=latency_ms,
                        encoder=encoder,
                        fact_similarity_threshold=fact_similarity_threshold,
                        input_price_per_million=input_price_per_million,
                        output_price_per_million=output_price_per_million,
                        provider_name=runner.provider_name,
                        model_name=runner.model_name,
                    )
                except Exception as exc:
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    provider_status = _provider_status_code(exc)
                    transient = _is_transient_provider_error(exc)
                    row = {
                        "evaluator_version": EVALUATOR_VERSION,
                        "question_id": question_id,
                        "question_type": normalize_question_type(question.get("question_type")),
                        "system": runner.name,
                        "llm_provider": runner.provider_name,
                        "llm_model": runner.model_name,
                        "latency_ms": round(latency_ms, 3),
                        "error_type": type(exc).__name__,
                        "provider_status_code": provider_status,
                        "retryable_error": transient,
                        "pair_attempt": pair_attempt,
                    }
                    append_result(results_path, row)
                    existing.append(row)
                    print(
                        f"Evaluated {question_id} with {runner.name}: "
                        f"{row['error_type']} "
                        f"(HTTP {provider_status or 'unknown'}, attempt "
                        f"{pair_attempt}/{max_pair_attempts})"
                    )
                    if provider_status == 429:
                        print(
                            "Provider quota remained exhausted after client retries; "
                            "evaluation stopped with progress saved."
                        )
                        return existing
                    if provider_status in {400, 401, 403, 404, 413, 424}:
                        print(
                            "Provider rejected the request with a non-retryable client error; "
                            "evaluation stopped with progress saved."
                        )
                        return existing
                    if transient and pair_attempt < max_pair_attempts:
                        if transient_cooldown_seconds > 0.0:
                            sleep(transient_cooldown_seconds)
                        continue
                    break
                else:
                    row["pair_attempt"] = pair_attempt
                    append_result(results_path, row)
                    existing.append(row)
                    completed.add(key)
                    print(f"Evaluated {question_id} with {runner.name}: {row['status']}")
                    break
    return existing


def resolve_model_name(provider_name: str, model_name: str | None = None) -> str:
    """Resolve the provider-specific evaluation model from CLI/configuration."""

    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider_name}")
    if model_name is not None and model_name.strip():
        return model_name.strip()

    settings = get_settings()
    if provider_name == "gemini":
        return settings.gemini_model
    return settings.groq_model


def resolve_pricing(
    *,
    provider_name: str,
    model_name: str,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> tuple[float, float]:
    """Resolve paid-tier token prices used only for evaluation cost estimates."""

    if (input_price_per_million is None) != (output_price_per_million is None):
        raise ValueError("input and output token prices must be supplied together")
    if input_price_per_million is not None and output_price_per_million is not None:
        if input_price_per_million < 0.0 or output_price_per_million < 0.0:
            raise ValueError("token prices must be non-negative")
        return input_price_per_million, output_price_per_million

    pricing = _PROVIDER_PRICING_USD_PER_MILLION.get((provider_name, model_name))
    if pricing is None:
        raise ValueError(
            "No default pricing is recorded for this provider/model. "
            "Pass both token-price arguments explicitly."
        )
    return pricing


def _artifact_slug(provider_name: str, model_name: str) -> str:
    raw = f"{provider_name}-{model_name}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    if not slug:
        raise ValueError("provider/model produced an empty artifact slug")
    return slug


def default_evaluation_paths(
    *,
    provider_name: str,
    model_name: str,
) -> tuple[Path, Path]:
    """Return provider-isolated artifact paths so model runs cannot contaminate each other."""

    slug = _artifact_slug(provider_name, model_name)
    root = Path("artifacts/evaluation")
    return (
        root / f"{slug}-answer-eval-results.jsonl",
        root / f"{slug}-answer-eval-summary.json",
    )


def build_runtime_runners(
    system_names: Sequence[str],
    *,
    provider_name: str = "gemini",
    model_name: str | None = None,
    context_sources: int | None = None,
    dense_context_sources: int | None = None,
    graph_context_sources: int | None = None,
    max_context_characters: int | None = None,
) -> tuple[list[AnswerRunner], FastEmbedTextEncoder]:
    """Construct requested systems over shared Qdrant, Neo4j, and one LLM provider."""

    unsupported = sorted(set(system_names) - set(DEFAULT_SYSTEMS))
    if unsupported:
        raise ValueError(f"Unsupported systems: {', '.join(unsupported)}")
    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider_name}")

    settings = get_settings()
    resolved_model = resolve_model_name(provider_name, model_name)
    if provider_name == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("Set EKA_GEMINI_API_KEY before running Gemini evaluation")
        language_model = GeminiRestClient(
            api_key=settings.gemini_api_key,
            model_name=resolved_model,
            base_url=settings.gemini_base_url,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
    else:
        if not settings.groq_api_key:
            raise RuntimeError("Set EKA_GROQ_API_KEY before running Groq evaluation")
        language_model = GroqRestClient(
            api_key=settings.groq_api_key,
            model_name=resolved_model,
            base_url=settings.groq_base_url,
            timeout_seconds=settings.groq_timeout_seconds,
        )

    resolved_context_sources = context_sources or settings.rag_context_sources
    resolved_dense_sources = dense_context_sources or settings.rag_dense_context_sources
    resolved_graph_sources = graph_context_sources or settings.rag_graph_context_sources
    resolved_max_characters = max_context_characters or settings.rag_max_context_characters
    if resolved_dense_sources + resolved_graph_sources > resolved_context_sources:
        raise ValueError(
            "dense_context_sources + graph_context_sources must not exceed context_sources"
        )

    encoder = FastEmbedTextEncoder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    qdrant = QdrantStore(
        base_url=settings.qdrant_url,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    qdrant.health()
    dense_retriever = VectorRetriever(
        store=qdrant,
        encoder=encoder,
        collection_name=settings.qdrant_collection,
    )
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    graph_store.verify_connectivity()
    graph_retriever = GraphRAGRetriever(
        dense_retriever=dense_retriever,
        graph_store=graph_store,
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
    graph_context = GraphContextBuilder(
        max_sources=resolved_context_sources,
        dense_sources=resolved_dense_sources,
        graph_sources=resolved_graph_sources,
        max_per_document=settings.rag_max_chunks_per_document,
        max_context_characters=resolved_max_characters,
    )
    dense_context = ContextBuilder(
        max_sources=resolved_context_sources,
        max_per_document=settings.rag_max_chunks_per_document,
        max_context_characters=resolved_max_characters,
    )
    dense_service = GroundedAnswerService(
        retriever=dense_retriever,
        language_model=language_model,
        context_builder=dense_context,
        retrieval_candidates=settings.rag_retrieval_candidates,
    )
    graph_service = GraphAugmentedAnswerService(
        retriever=dense_retriever,
        graph_retriever=graph_retriever,
        language_model=language_model,
        context_builder=graph_context,
        retrieval_candidates=settings.rag_retrieval_candidates,
        graph_document_candidates=settings.rag_graph_document_candidates,
        graph_fetch_candidates=settings.rag_graph_fetch_candidates,
        min_graph_matched_entities=settings.rag_graph_min_matched_entities,
    )
    tracer = build_tracer(
        enabled=settings.mlflow_enabled,
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )
    agent = EnterpriseKnowledgeAgent(
        planner=language_model,
        dense_retriever=dense_retriever,
        graph_retriever=graph_retriever,
        answer_service=dense_service,
        context_builder=graph_context,
        retrieval_candidates=settings.rag_retrieval_candidates,
        graph_document_candidates=settings.rag_graph_document_candidates,
        graph_fetch_candidates=settings.rag_graph_fetch_candidates,
        min_graph_matched_entities=settings.rag_graph_min_matched_entities,
        max_tool_calls=settings.agent_max_tool_calls,
        tracer=tracer,
    )

    available: dict[str, AnswerRunner] = {
        "dense": GroundedAnswerRunner(
            name="dense",
            service=dense_service,
            provider_name=provider_name,
            model_name=resolved_model,
        ),
        "graph": GroundedAnswerRunner(
            name="graph",
            service=graph_service,
            provider_name=provider_name,
            model_name=resolved_model,
        ),
        "agent": AgentAnswerRunner(
            agent=agent,
            provider_name=provider_name,
            model_name=resolved_model,
        ),
    }
    return [available[name] for name in system_names], encoder


def log_summary_to_mlflow(summary: dict[str, Any]) -> None:
    """Best-effort logging of aggregate, privacy-safe evaluation metrics to MLflow."""

    settings = get_settings()
    if not settings.mlflow_enabled:
        return
    try:
        import mlflow
    except ImportError:
        print("MLflow logging skipped: optional ops dependency is unavailable")
        return

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name="answer-evaluation"):
            mlflow.log_params(
                {
                    "evaluator_version": summary["evaluator_version"],
                    "sample_per_type": summary["sample_per_type"],
                    "sample_seed": summary["sample_seed"],
                    "selected_question_count": summary["selected_question_count"],
                    "systems": ",".join(summary["systems"]),
                    "leaderboard_comparable": summary["leaderboard_comparable"],
                }
            )
            mlflow.log_param("evaluation_complete", summary["evaluation_complete"])
            for system_name, metrics in summary["metrics_by_system"].items():
                mlflow.log_metric(
                    f"{system_name}.error_count", float(metrics.get("error_count", 0))
                )
                if not summary["evaluation_complete"]:
                    continue
                for field in (
                    "answerability_accuracy",
                    "mean_document_recall",
                    "mean_citation_precision",
                    "mean_gold_answer_similarity_proxy",
                    "mean_answer_fact_coverage_proxy",
                    "mean_latency_ms",
                    "mean_total_tokens",
                    "estimated_standard_paid_cost_usd",
                    "planner_policy_alignment",
                    "planner_fallback_rate",
                    "graph_tool_call_rate",
                    "graph_tool_yield_rate",
                ):
                    value = metrics.get(field)
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{system_name}.{field}", float(value))
            mlflow.log_dict(summary, "answer_evaluation_summary.json")
    except Exception as exc:
        print(f"MLflow aggregate logging skipped after evaluation: {type(exc).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate dense RAG, graph context, and the LangGraph agent."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/processed/enterprise_rag_bench/benchmark_questions.jsonl"),
    )
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample-per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=DEFAULT_SYSTEMS,
        default=list(DEFAULT_SYSTEMS),
    )
    parser.add_argument("--fact-similarity-threshold", type=float, default=0.70)
    parser.add_argument("--input-price-per-million", type=float, default=None)
    parser.add_argument("--output-price-per-million", type=float, default=None)
    parser.add_argument("--pause-seconds", type=float, default=None)
    parser.add_argument("--transient-cooldown-seconds", type=float, default=None)
    parser.add_argument("--max-pair-attempts", type=int, default=2)
    parser.add_argument("--context-sources", type=int, default=None)
    parser.add_argument("--dense-context-sources", type=int, default=None)
    parser.add_argument("--graph-context-sources", type=int, default=None)
    parser.add_argument("--max-context-characters", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    provider_name = args.provider
    model_name = resolve_model_name(provider_name, args.model)
    input_price, output_price = resolve_pricing(
        provider_name=provider_name,
        model_name=model_name,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    default_results, default_summary = default_evaluation_paths(
        provider_name=provider_name,
        model_name=model_name,
    )
    results_path = args.results or default_results
    summary_path = args.summary or default_summary

    context_sources = args.context_sources or settings.rag_context_sources
    dense_context_sources = args.dense_context_sources or settings.rag_dense_context_sources
    graph_context_sources = args.graph_context_sources or settings.rag_graph_context_sources
    max_context_characters = args.max_context_characters or settings.rag_max_context_characters
    pause_seconds = args.pause_seconds
    if pause_seconds is None:
        pause_seconds = 35.0 if provider_name == "groq" else 8.0
    transient_cooldown_seconds = args.transient_cooldown_seconds
    if transient_cooldown_seconds is None:
        transient_cooldown_seconds = 30.0 if provider_name == "groq" else 45.0

    questions = select_balanced_questions(
        read_jsonl(args.questions),
        sample_per_type=args.sample_per_type,
        seed=args.seed,
    )
    runners, semantic_encoder = build_runtime_runners(
        args.systems,
        provider_name=provider_name,
        model_name=model_name,
        context_sources=context_sources,
        dense_context_sources=dense_context_sources,
        graph_context_sources=graph_context_sources,
        max_context_characters=max_context_characters,
    )
    rows = evaluate_systems(
        questions=questions,
        runners=runners,
        encoder=semantic_encoder,
        results_path=results_path,
        fact_similarity_threshold=args.fact_similarity_threshold,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
        pause_seconds=pause_seconds,
        transient_cooldown_seconds=transient_cooldown_seconds,
        max_pair_attempts=args.max_pair_attempts,
    )
    summary = summarize_rows(
        rows=rows,
        selected_questions=questions,
        systems=args.systems,
        sample_per_type=args.sample_per_type,
        seed=args.seed,
        fact_similarity_threshold=args.fact_similarity_threshold,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
        provider_name=provider_name,
        model_name=model_name,
        context_sources=context_sources,
        dense_context_sources=dense_context_sources,
        graph_context_sources=graph_context_sources,
        max_context_characters=max_context_characters,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log_summary_to_mlflow(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["evaluation_complete"]:
        raise SystemExit(
            "Evaluation incomplete: provider failures remain. Re-run the same command; "
            "successful pairs are skipped and failed pairs are retried."
        )


if __name__ == "__main__":
    main()
