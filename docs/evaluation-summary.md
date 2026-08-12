# Evaluation Summary

This document collects the main committed measurements in one place. It is an engineering summary of the experiments in this repository, not an EnterpriseRAG-Bench leaderboard submission.

## Corpus and graph scale

| Item | Value |
|---|---:|
| Normalized documents | 19,361 |
| Qdrant chunks | 62,316 |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Canonical entities | 47,787 |
| Neo4j `MENTIONS` relationships | 121,030 |
| Neo4j `CO_OCCURS_WITH` relationships | 331,933 |

Sources:

- `artifacts/retrieval/vector_index_stats.json`
- `artifacts/nlp/entity_extraction_stats.json`
- `artifacts/graph/graph_build_stats.json`
- `artifacts/graph/graph_verification.json`

## Dense retrieval baseline

The dense Qdrant baseline was evaluated on 177 compatible questions. Twenty benchmark questions without usable gold documents were excluded from retrieval scoring.

| Metric | Result |
|---|---:|
| Hit Rate@1 | 0.469 |
| Hit Rate@10 | 0.661 |
| Recall@1 | 0.373 |
| Recall@10 | 0.603 |
| MRR@10 | 0.534 |
| Mean query latency | 53.8 ms |

The full overall and per-question-type results are in `artifacts/retrieval/vector_baseline_metrics.json`.

## Graph-ranking experiment

Graph-assisted retrieval seeded Neo4j expansion from dense results and combined the dense and graph rankings with reciprocal rank fusion.

| Metric | Dense | Graph fusion | Delta |
|---|---:|---:|---:|
| Hit Rate@1 | 0.469 | 0.305 | -0.164 |
| Hit Rate@10 | 0.661 | 0.667 | +0.006 |
| Recall@1 | 0.373 | 0.231 | -0.141 |
| Recall@10 | 0.603 | 0.607 | +0.004 |
| MRR@10 | 0.534 | 0.414 | -0.120 |
| Mean query latency | 53.8 ms | 135.3 ms | +81.5 ms |

The graph path recovered some additional deep-rank evidence, including gains for a small subset of question types, but it substantially degraded early-rank quality and increased latency.

The runtime therefore does not use graph fusion as the primary ranking method. Dense retrieval remains authoritative and graph evidence is admitted only as a bounded context supplement.

The complete comparison is in `artifacts/retrieval/graphrag_comparison.json`.

## Answer and agent comparison

A controlled local comparison used:

- 18 deterministically selected questions;
- two questions from each of nine compatible question types;
- three systems: dense RAG, graph-context RAG, and the LangGraph agent;
- the same Groq `openai/gpt-oss-20b` model for every compared system;
- the same 8,000-character context ceiling;
- 54/54 successful system/question outputs.

The metrics below are local regression proxies. They are not official benchmark correctness/completeness scores.

| Metric | Dense | Graph context | Agent |
|---|---:|---:|---:|
| Answerability accuracy | 0.333 | 0.333 | 0.167 |
| Gold-answer similarity proxy | 0.227 | 0.216 | 0.056 |
| Answer-fact coverage proxy | 0.193 | 0.098 | 0.063 |
| Citation precision | 0.125 | 0.094 | 0.063 |
| Expected-document recall | 0.094 | 0.083 | 0.063 |
| Mean invalid extra docs | 0.250 | 0.438 | 0.000 |
| Mean latency | 836 ms | 1,129 ms | 1,484 ms |
| Mean total tokens | 2,525 | 2,511 | 2,855 |

For the agent:

| Agent diagnostic | Result |
|---|---:|
| Planner-policy alignment | 0.389 |
| Planner fallback rate | 0.000 |
| Mean retrieval tool calls | 1.833 |
| Graph-tool call rate | 0.833 |
| Graph-tool yield rate | 0.667 |

All three systems achieved 1.0 answerability accuracy on the two `info_not_found` questions in this small sample.

### Interpretation

The answer experiment does not support a claim that the graph or agent path improves aggregate answer quality. Under this model and context budget, dense RAG had the strongest local semantic/citation proxies and the lowest latency. The agent routed to graph expansion frequently and paid additional latency/token cost without improving the aggregate results.

The experiment is deliberately retained instead of being tuned away. It provides evidence for a simpler default retrieval strategy and identifies concrete areas for future work: planner calibration, better graph entity resolution, graph-context selection, stronger answer models, and evaluation on a larger held-out set.

The complete summary and per-question privacy-safe rows are:

- `artifacts/evaluation/groq-openai-gpt-oss-20b-answer-eval-summary.json`
- `artifacts/evaluation/groq-openai-gpt-oss-20b-answer-eval-results.jsonl`

## Evaluation boundaries

- The corpus is benchmark data rather than a live enterprise deployment.
- The answer comparison uses a small balanced subset to keep provider usage manageable.
- Local embedding-based answer metrics are regression proxies, not judge-based benchmark scores.
- Retrieval and answer experiments measure different layers and should not be compared as if their metrics were interchangeable.
- Graph co-occurrence is an association signal; it is not treated as verified causation or ownership.
