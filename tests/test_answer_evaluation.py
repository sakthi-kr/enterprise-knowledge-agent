"""Tests for privacy-safe answer evaluation and local semantic proxy metrics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from enterprise_knowledge_agent.agent_types import AgentStrategy
from enterprise_knowledge_agent.answer_evaluation import (
    EVALUATOR_VERSION,
    EvaluationOutput,
    build_result_row,
    cosine_similarity,
    default_evaluation_paths,
    estimate_standard_paid_cost_usd,
    evaluate_systems,
    expected_agent_strategy,
    resolve_pricing,
    select_balanced_questions,
    semantic_answer_proxies,
    summarize_rows,
)
from enterprise_knowledge_agent.gemini_client import GeminiAPIError
from enterprise_knowledge_agent.groq_client import GroqAPIError
from enterprise_knowledge_agent.grounded_answer import (
    AnswerStatus,
    EvidenceSource,
    GroundedAnswer,
    TokenUsage,
)


class _Encoder:
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "candidate answer": [1.0, 0.0],
            "gold answer": [1.0, 0.0],
            "fact one": [1.0, 0.0],
            "fact two": [0.0, 1.0],
        }
        return [vectors.get(text, [1.0, 0.0]) for text in texts]


def _citation(doc_id: str) -> EvidenceSource:
    return EvidenceSource(
        citation_id="S1",
        rank=1,
        score=0.9,
        chunk_id="chunk-1",
        record_id="record-1",
        doc_id=doc_id,
        source_type="jira",
        title="Incident",
        source_file="jira/item.txt",
        text="Evidence",
    )


def _answer(*, status: AnswerStatus = AnswerStatus.ANSWERED) -> GroundedAnswer:
    citations = (_citation("doc-a"),) if status is AnswerStatus.ANSWERED else ()
    return GroundedAnswer(
        status=status,
        answer="candidate answer",
        citations=citations,
        model_name="model",
        usage=TokenUsage(prompt_tokens=100, output_tokens=20, total_tokens=120),
        retrieved_chunk_count=12,
        context_source_count=6,
    )


def _question(question_id: str, question_type: str = "basic") -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": f"Question {question_id}?",
        "expected_doc_ids": ["doc-a"],
        "gold_answer": "gold answer",
        "answer_facts": ["fact one"],
    }


def test_balanced_selection_is_deterministic_and_balanced() -> None:
    questions = [
        _question("q1", "basic"),
        _question("q2", "basic"),
        _question("q3", "basic"),
        _question("q4", "semantic"),
        _question("q5", "semantic"),
        _question("q6", "semantic"),
    ]
    first = select_balanced_questions(questions, sample_per_type=2, seed=7)
    second = select_balanced_questions(reversed(questions), sample_per_type=2, seed=7)
    assert [row["question_id"] for row in first] == [row["question_id"] for row in second]
    assert len(first) == 4


def test_expected_agent_strategy_uses_cross_document_policy() -> None:
    assert expected_agent_strategy(_question("q1")) is AgentStrategy.DENSE_ONLY
    assert (
        expected_agent_strategy(_question("q2", "project_related"))
        is AgentStrategy.DENSE_PLUS_GRAPH
    )
    multi = _question("q3")
    multi["expected_doc_ids"] = ["doc-a", "doc-b"]
    assert expected_agent_strategy(multi) is AgentStrategy.DENSE_PLUS_GRAPH


def test_cosine_similarity_and_semantic_proxies() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    similarity, coverage = semantic_answer_proxies(
        answer="candidate answer",
        gold_answer="gold answer",
        answer_facts=["fact one", "fact two"],
        encoder=_Encoder(),
        fact_similarity_threshold=0.8,
    )
    assert similarity == pytest.approx(1.0)
    assert coverage == pytest.approx(0.5)


def test_paid_cost_includes_planner_and_thinking_tokens() -> None:
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        output_tokens=100_000,
        thinking_tokens=100_000,
        total_tokens=1_200_000,
    )
    assert estimate_standard_paid_cost_usd(
        usage,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    ) == pytest.approx(3.0)


def test_result_row_keeps_only_safe_benchmark_metadata() -> None:
    question = _question("q1")
    output = EvaluationOutput(answer=_answer())
    row = build_result_row(
        question=question,
        system_name="dense",
        output=output,
        latency_ms=25.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    serialized = str(row)
    assert row["answerability_correct"] is True
    assert row["document_recall"] == 1.0
    assert row["citation_precision"] == 1.0
    assert "Question q1?" not in serialized
    assert "gold answer" not in serialized
    assert "candidate answer" not in serialized
    assert "fact one" not in serialized
    assert "Evidence" not in serialized


def test_info_not_found_scores_abstention_without_semantic_reference() -> None:
    question = _question("qnf", "info not found")
    question["expected_doc_ids"] = []
    output = EvaluationOutput(answer=_answer(status=AnswerStatus.INSUFFICIENT_EVIDENCE))
    row = build_result_row(
        question=question,
        system_name="dense",
        output=output,
        latency_ms=10.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    assert row["answerability_correct"] is True
    assert row["gold_answer_similarity_proxy"] is None
    assert row["answer_fact_coverage_proxy"] is None


def test_agent_row_includes_policy_and_total_usage() -> None:
    question = _question("qagent", "project related")
    output = EvaluationOutput(
        answer=_answer(),
        planner_strategy="dense_plus_graph",
        planner_fallback=False,
        tool_call_count=2,
        graph_tool_called=True,
        graph_tool_result_count=1,
        planner_usage=TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=15),
    )
    row = build_result_row(
        question=question,
        system_name="agent",
        output=output,
        latency_ms=50.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    assert row["planner_policy_alignment"] is True
    assert row["total_tokens"] == 135
    assert row["graph_tool_called"] is True


def test_summary_compares_candidate_systems_against_dense() -> None:
    question = _question("q1")
    dense = build_result_row(
        question=question,
        system_name="dense",
        output=EvaluationOutput(answer=_answer()),
        latency_ms=20.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    graph_answer = replace(
        _answer(),
        usage=TokenUsage(prompt_tokens=120, output_tokens=20, total_tokens=140),
    )
    graph = build_result_row(
        question=question,
        system_name="graph",
        output=EvaluationOutput(answer=graph_answer),
        latency_ms=30.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    summary = summarize_rows(
        rows=[dense, graph],
        selected_questions=[question],
        systems=["dense", "graph"],
        sample_per_type=1,
        seed=17,
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    assert summary["evaluator_version"] == EVALUATOR_VERSION
    assert summary["leaderboard_comparable"] is False
    assert summary["comparison_vs_dense"]["graph"]["mean_latency_ms"]["delta"] == 10.0


class _FlakyRunner:
    name = "dense"
    provider_name = "gemini"
    model_name = "gemini-3.6-flash"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, question: str) -> EvaluationOutput:
        self.calls += 1
        if self.calls == 1:
            raise GeminiAPIError("Gemini API returned HTTP 503: temporarily unavailable")
        return EvaluationOutput(answer=_answer())


def test_evaluation_retries_failed_pair_and_latest_success_wins(tmp_path) -> None:
    question = _question("q-retry")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        '{"evaluator_version":"2026-08-local-semantic-v1",'
        '"question_id":"q-retry","question_type":"basic",'
        '"system":"dense","error_type":"GeminiAPIError"}\n',
        encoding="utf-8",
    )
    runner = _FlakyRunner()
    rows = evaluate_systems(
        questions=[question],
        runners=[runner],
        encoder=_Encoder(),
        results_path=results_path,
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
        pause_seconds=0.0,
        transient_cooldown_seconds=0.0,
        max_pair_attempts=2,
        sleep=lambda _seconds: None,
    )
    assert runner.calls == 2
    assert rows[-2]["provider_status_code"] == 503
    assert rows[-2]["retryable_error"] is True
    assert rows[-1]["status"] == "answered"

    summary = summarize_rows(
        rows=rows,
        selected_questions=[question],
        systems=["dense"],
        sample_per_type=1,
        seed=17,
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    assert summary["evaluation_complete"] is True
    assert summary["metrics_by_system"]["dense"]["error_count"] == 0
    assert summary["metrics_by_system"]["dense"]["successful_question_count"] == 1


def test_incomplete_summary_suppresses_cross_system_comparison() -> None:
    question = _question("q-incomplete")
    dense = build_result_row(
        question=question,
        system_name="dense",
        output=EvaluationOutput(answer=_answer()),
        latency_ms=20.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    graph_error = {
        "evaluator_version": EVALUATOR_VERSION,
        "question_id": "q-incomplete",
        "question_type": "basic",
        "system": "graph",
        "error_type": "GeminiAPIError",
    }
    summary = summarize_rows(
        rows=[dense, graph_error],
        selected_questions=[question],
        systems=["dense", "graph"],
        sample_per_type=1,
        seed=17,
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
    )
    assert summary["evaluation_complete"] is False
    assert summary["comparison_vs_dense"] == {}
    assert summary["metrics_by_system"]["graph"]["error_count"] == 1


def test_groq_pricing_and_artifact_paths_are_provider_isolated() -> None:
    assert resolve_pricing(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        input_price_per_million=None,
        output_price_per_million=None,
    ) == (0.075, 0.30)
    results, summary = default_evaluation_paths(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
    )
    assert results.name == "groq-openai-gpt-oss-20b-answer-eval-results.jsonl"
    assert summary.name == "groq-openai-gpt-oss-20b-answer-eval-summary.json"


class _QuotaRunner:
    name = "dense"
    provider_name = "groq"
    model_name = "openai/gpt-oss-20b"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, question: str) -> EvaluationOutput:
        self.calls += 1
        raise GroqAPIError("Groq API returned HTTP 429: token limit")


def test_persistent_429_stops_evaluation_immediately(tmp_path) -> None:
    runner = _QuotaRunner()
    rows = evaluate_systems(
        questions=[_question("q1"), _question("q2")],
        runners=[runner],
        encoder=_Encoder(),
        results_path=tmp_path / "groq.jsonl",
        fact_similarity_threshold=0.7,
        input_price_per_million=0.075,
        output_price_per_million=0.30,
        pause_seconds=0.0,
        transient_cooldown_seconds=0.0,
        max_pair_attempts=2,
        sleep=lambda _seconds: None,
    )
    assert runner.calls == 1
    assert len(rows) == 1
    assert rows[0]["provider_status_code"] == 429
    assert rows[0]["llm_provider"] == "groq"


class _AuthFailureRunner:
    name = "dense"
    provider_name = "groq"
    model_name = "openai/gpt-oss-20b"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, question: str) -> EvaluationOutput:
        self.calls += 1
        raise GroqAPIError("Groq API returned HTTP 401: invalid key")


def test_non_retryable_provider_error_stops_evaluation_immediately(tmp_path) -> None:
    runner = _AuthFailureRunner()
    rows = evaluate_systems(
        questions=[_question("q1"), _question("q2")],
        runners=[runner],
        encoder=_Encoder(),
        results_path=tmp_path / "groq.jsonl",
        fact_similarity_threshold=0.7,
        input_price_per_million=0.075,
        output_price_per_million=0.30,
        pause_seconds=0.0,
        transient_cooldown_seconds=0.0,
        max_pair_attempts=2,
        sleep=lambda _seconds: None,
    )
    assert runner.calls == 1
    assert len(rows) == 1
    assert rows[0]["provider_status_code"] == 401
    assert rows[0]["retryable_error"] is False


def test_summary_filters_provider_specific_runs() -> None:
    question = _question("q-provider")
    gemini = build_result_row(
        question=question,
        system_name="dense",
        output=EvaluationOutput(answer=_answer()),
        latency_ms=20.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=1.5,
        output_price_per_million=7.5,
        provider_name="gemini",
        model_name="gemini-3.6-flash",
    )
    groq = build_result_row(
        question=question,
        system_name="dense",
        output=EvaluationOutput(answer=_answer()),
        latency_ms=30.0,
        encoder=_Encoder(),
        fact_similarity_threshold=0.7,
        input_price_per_million=0.075,
        output_price_per_million=0.30,
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
    )
    summary = summarize_rows(
        rows=[gemini, groq],
        selected_questions=[question],
        systems=["dense"],
        sample_per_type=1,
        seed=17,
        fact_similarity_threshold=0.7,
        input_price_per_million=0.075,
        output_price_per_million=0.30,
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        context_sources=4,
        dense_context_sources=3,
        graph_context_sources=1,
        max_context_characters=8000,
    )
    assert summary["evaluation_complete"] is True
    assert summary["evaluation_provider"] == "groq"
    assert summary["metrics_by_system"]["dense"]["mean_latency_ms"] == 30.0
    assert summary["context_budget"]["max_sources"] == 4
