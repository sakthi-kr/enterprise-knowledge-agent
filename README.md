# Enterprise Knowledge Agent

[![CI](https://github.com/sakthi-kr/enterprise-knowledge-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sakthi-kr/enterprise-knowledge-agent/actions/workflows/ci.yml)

An evaluated enterprise RAG, GraphRAG, and agent system built over multi-source enterprise-style data. The project combines Qdrant vector retrieval, a Neo4j knowledge graph, grounded LLM answers, bounded LangGraph orchestration, privacy-safe MLflow tracing, and a production-style FastAPI service.

The repository is intentionally evidence-driven: graph retrieval and agent orchestration are measured against simpler baselines rather than assumed to be better.

## What is implemented

- Reproducible ingestion of Confluence-, Jira-, and GitHub-style records from EnterpriseRAG-Bench.
- Deterministic document/chunk identifiers and source provenance.
- Local BGE embeddings with Qdrant dense-vector retrieval.
- Retrieval evaluation against benchmark evidence documents.
- Grounded LLM answers with validated inline citations and explicit insufficient-evidence handling.
- Local GLiNER2 entity extraction for people, organizations, teams, projects, services, technologies, and repositories.
- Neo4j graph construction with `MENTIONS` and evidence-backed document co-occurrence relationships.
- An evaluated graph-retrieval experiment and a dense-first, bounded graph-context runtime.
- A LangGraph agent with explicit state, conditional dense/graph tool routing, a hard tool-call limit, and failure fallbacks.
- Gemini as the application LLM provider, with Groq support in the controlled evaluation harness.
- Privacy-safe MLflow traces that retain operational metadata without copying questions or enterprise text into the tracing store.
- FastAPI liveness/readiness endpoints, request IDs, structured logs, stable error envelopes, and timeouts.
- Docker and Docker Compose packaging for the API, Qdrant, Neo4j, and MLflow.
- CI for Ruff linting/formatting and the full pytest suite.

## Architecture

```text
EnterpriseRAG-Bench
Confluence / Jira / GitHub
          |
          v
  normalization + chunking
          |
     +----+------------------------+
     |                             |
     v                             v
BGE embeddings                 GLiNER2
     |                             |
     v                             v
  Qdrant                    canonical entities
     |                             |
     |                             v
     |                           Neo4j
     |                             |
     +-------------+---------------+
                   |
                 FastAPI
              /ask      /agent/ask
                |            |
       dense-first RAG   LangGraph planner
                |        dense -> optional graph
                +------------+
                     |
              grounded context
                     |
                  Gemini
                     |
          answer + validated citations
                     |
          optional privacy-safe MLflow
```

The application `/ask` path keeps dense ranking authoritative and allows only a small graph-derived context supplement. The `/agent/ask` path lets the planner choose dense retrieval alone or one bounded graph expansion after dense retrieval. See [docs/architecture.md](docs/architecture.md) for the component boundaries and runtime flow.

## Measured scale

The committed artifacts describe a locally built corpus and graph:

| Component | Measured value |
|---|---:|
| Normalized documents | 19,361 |
| Indexed chunks | 62,316 |
| Embedding dimension | 384 |
| Canonical graph entities | 47,787 |
| `MENTIONS` relationships | 121,030 |
| `CO_OCCURS_WITH` relationships | 331,933 |

These figures come from the committed summaries under `artifacts/retrieval/`, `artifacts/nlp/`, and `artifacts/graph/`.

## Retrieval results

Dense retrieval was evaluated on 177 compatible benchmark questions:

| Metric | Dense baseline |
|---|---:|
| Hit Rate@10 | 0.661 |
| Recall@10 | 0.603 |
| MRR@10 | 0.534 |
| Mean query latency | 53.8 ms |

A graph-assisted reciprocal-rank-fusion experiment slightly increased rank-10 coverage but damaged early-rank quality:

| Metric | Dense | Graph fusion | Delta |
|---|---:|---:|---:|
| Hit Rate@10 | 0.661 | 0.667 | +0.006 |
| Recall@10 | 0.603 | 0.607 | +0.004 |
| MRR@10 | 0.534 | 0.414 | -0.120 |
| Mean latency | 53.8 ms | 135.3 ms | +81.5 ms |

That result changed the runtime design: graph evidence is used as a bounded context supplement instead of being allowed to reorder the strongest dense results.

A separate 54-output comparison of dense RAG, graph-context RAG, and the LangGraph agent also found that the simpler dense path had the strongest aggregate local proxy metrics and the lowest latency under the tested Groq `openai/gpt-oss-20b` configuration. The experiment is intentionally a small internal regression comparison, not an EnterpriseRAG-Bench leaderboard score. See [docs/evaluation-summary.md](docs/evaluation-summary.md) for the full numbers and limitations.

## Quick development setup

Python 3.10 is the target version.

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

Optional dependencies:

```bash
python -m pip install -e ".[dev,nlp,graph,ops]"
```

Copy `.env.example` to `.env` and add the provider credentials you intend to use. `.env` is excluded from Git and from the Docker build context.

## Rebuild the corpus and indexes

Download and prepare the benchmark subset:

```bash
python -m enterprise_knowledge_agent.dataset_download
python -m enterprise_knowledge_agent.data_pipeline
```

Build the vector index:

```bash
docker compose up -d qdrant
python -m enterprise_knowledge_agent.vector_index --recreate
```

Extract entities and build the graph:

```bash
python -m enterprise_knowledge_agent.entity_extraction
docker compose up -d neo4j
python -m enterprise_knowledge_agent.graph_build
python -m enterprise_knowledge_agent.graph_verify
```

The raw benchmark archives, processed corpus, vector database, and Neo4j database are intentionally not committed. Small reproducible experiment summaries are committed under `artifacts/`.

## Run the application

With Qdrant and Neo4j populated and provider configuration present:

```bash
python -m uvicorn enterprise_knowledge_agent.main:app --reload
```

Main endpoints:

```text
GET  /health
GET  /ready
POST /ask
POST /agent/ask
```

Example grounded request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What caused the API gateway autoscaler incident?"}'
```

The response includes the answer status, validated citations, provider token usage, retrieval strategy, and graph-context diagnostics.

## Run the local service stack

After the Qdrant index and Neo4j graph have been built, start the API, Qdrant, Neo4j, and MLflow together:

```bash
docker compose up -d --build --wait --wait-timeout 180
bash scripts/smoke_stack.sh
```

The smoke test verifies Qdrant readiness, a real Neo4j query, MLflow health, API liveness, and API dependency readiness without making an LLM call.

Local endpoints:

- API docs: `http://127.0.0.1:8000/docs`
- MLflow: `http://127.0.0.1:5000`
- Neo4j Browser: `http://127.0.0.1:7474`
- Qdrant: `http://127.0.0.1:6333`

Stop the stack without deleting the persistent named volumes:

```bash
docker compose down
```

## Evaluation

Retrieval:

```bash
python -m enterprise_knowledge_agent.retrieval_evaluation
python -m enterprise_knowledge_agent.graphrag_evaluation
```

Answer/agent comparison:

```bash
python -m enterprise_knowledge_agent.answer_evaluation --provider groq \
  --sample-per-type 2 \
  --context-sources 4 \
  --dense-context-sources 3 \
  --graph-context-sources 1 \
  --max-context-characters 8000
```

Provider/model runs are stored separately, successful pairs are resumable, and incomplete runs do not emit cross-system comparisons. Benchmark questions, gold answers, generated answers, and retrieved text are omitted from committed answer-evaluation artifacts.

## Key design decisions

**Dense evidence stays in control.** The graph-fusion experiment improved only deep-rank coverage while substantially reducing MRR and increasing latency, so the runtime uses graph evidence only as a bounded supplement.

**Graph edges are evidence-backed associations, not invented semantics.** The graph records document mentions and document-level co-occurrence. It does not relabel co-occurrence as causation, ownership, or dependency.

**Agent execution is bounded.** The planner can choose dense retrieval or dense plus one graph expansion, with a hard maximum of two retrieval tool calls and fallback behavior when planning or graph access fails.

**Observability does not copy enterprise content.** MLflow receives span structure, timings, token counts, routing metadata, opaque IDs, and scores, while raw questions, retrieved text, source paths, planner explanations, and generated answer bodies remain out of traces.

## Limitations

- The corpus is enterprise-style benchmark data, not a live company knowledge base.
- The API runtime currently uses Gemini; Groq is implemented for controlled evaluation rather than as the normal application provider.
- The graph is based on entity mentions and co-occurrence, so relationship quality depends on extraction quality and does not encode verified business semantics.
- The agent did not outperform the simpler dense baseline in the current small answer-quality experiment; it is retained as an evaluated orchestration capability, not presented as a universal improvement.
- The answer-quality experiment uses local semantic proxy metrics on a balanced 18-question subset and is not leaderboard-comparable.
- The local stack has no authentication, tenant isolation, ACL-aware retrieval, or internet-facing hardening. It is intended for local development and portfolio demonstration.
- The vector index, graph database, and full processed corpus are reproducible local artifacts and are not stored in Git.

## Repository map

```text
src/enterprise_knowledge_agent/   application and evaluation code
tests/                            automated tests
docs/                             architecture and design notes
artifacts/retrieval/              retrieval measurements
artifacts/nlp/                    entity-extraction summary
artifacts/graph/                  graph build/verification summaries
artifacts/evaluation/             answer/agent comparison artifacts
data/                             local data layout documentation
scripts/                          local stack smoke tests
```

Useful documentation:

- [Architecture](docs/architecture.md)
- [Evaluation summary](docs/evaluation-summary.md)
- [Agent orchestration](docs/agent-orchestration.md)
- [GraphRAG runtime decision](docs/graphrag-context.md)
- [Answer evaluation](docs/answer-evaluation.md)
- [MLflow observability](docs/llm-observability.md)
- [Production runtime](docs/production-runtime.md)
- [Local service stack](docs/local-stack.md)
- [Completion criteria](docs/acceptance-criteria.md)
