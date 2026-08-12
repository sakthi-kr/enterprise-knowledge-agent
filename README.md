# Enterprise Knowledge Agent

A Python project for building and evaluating an enterprise knowledge assistant that combines vector retrieval, graph retrieval, grounded LLM generation, and bounded agent orchestration over multi-source enterprise data.

## Current implementation

The repository contains a FastAPI application, environment-based configuration, automated quality checks, a reproducible EnterpriseRAG-Bench data pipeline, dense-vector retrieval, grounded answer generation, local enterprise entity extraction, a Neo4j knowledge graph, graph-augmented answer context, and a LangGraph agent workflow.

The dense retrieval baseline uses a local FastEmbed model for CPU inference and Qdrant for persistent vector search, with deterministic chunk provenance and benchmark evaluation against ground-truth document IDs. Grounded answers use a Gemini provider adapter with structured output, validated inline citations, insufficient-evidence handling, and retry logic for transient provider failures.

Entity extraction uses a local GLiNER2 model with descriptive labels to identify stable named enterprise entities such as people, organizations, teams, projects, services, technologies, and repositories. Entity mentions are normalized into stable canonical IDs.

Neo4j stores source documents and canonical entities with evidence-backed `MENTIONS` relationships and document-level `CO_OCCURS_WITH` relationships. The graph keeps association separate from stronger semantic claims such as ownership or causation.

An evaluated reciprocal-rank-fusion experiment found that graph expansion produced a small improvement at rank 10 but degraded early-rank precision and MRR. The runtime answer path therefore keeps dense retrieval authoritative and uses graph retrieval as a bounded context supplement rather than allowing noisy graph evidence to reorder the strongest dense results. The measured comparison is retained under `artifacts/retrieval/`.

The agent workflow uses LangGraph with explicit state and bounded control flow. Gemini selects either dense retrieval alone or dense retrieval followed by graph expansion. Tool execution is recorded, graph-tool failure falls back to dense evidence, and the workflow has a hard tool-call limit before grounded answer synthesis.

Optional MLflow tracing records the agent request as a hierarchical trace with planner, dense retrieval, graph expansion, and grounded synthesis spans. Traces keep model, token, latency, routing, retrieval-score, and outcome metadata while redacting raw questions, retrieved text, source paths, and generated answer bodies. The answer-evaluation harness can run all compared systems against one selected provider/model while keeping benchmark references local.

## Development

Python 3.10 is the target development version.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Run the API locally with:

```bash
python -m uvicorn enterprise_knowledge_agent.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health`.

## Prepare the enterprise corpus

Download and validate the recommended Confluence, Jira, and GitHub subset:

```bash
python -m enterprise_knowledge_agent.dataset_download
```

Then normalize the documents, create deterministic overlapping chunks, and select benchmark questions that are valid for the local corpus:

```bash
python -m enterprise_knowledge_agent.data_pipeline
```

Generated corpus files are written to `data/processed/enterprise_rag_bench/` and are excluded from Git.

## Build and evaluate vector retrieval

Start the local Qdrant service:

```bash
docker compose up -d qdrant
```

Build a fresh dense-vector index from the prepared chunks:

```bash
python -m enterprise_knowledge_agent.vector_index --recreate
```

Search it directly:

```bash
python -m enterprise_knowledge_agent.vector_search "What caused the API gateway incident?"
```

Evaluate retrieval against benchmark document IDs:

```bash
python -m enterprise_knowledge_agent.retrieval_evaluation
```

Small experiment outputs are written to `artifacts/retrieval/`.

## Ask grounded questions

Copy `.env.example` to `.env` and set `EKA_GEMINI_API_KEY` to a Gemini API key. The `.env` file is excluded from Git.

Keep Qdrant and Neo4j running, then ask a question from the command line:

```bash
python -m enterprise_knowledge_agent.rag_query "What caused the API gateway autoscaler incident?"
```

Or start the API and send a request to `POST /ask`:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What caused the API gateway autoscaler incident?"}'
```

The response includes validated citations, source provenance, provider token usage, the retrieval strategy, and the number of graph-derived context sources actually supplied to the language model.

## Run the agent

Keep Qdrant and Neo4j running and ensure the Gemini API key is configured. Run one agent query from the command line:

```bash
python -m enterprise_knowledge_agent.agent_query \
  "How are the API gateway and autoscaler related?"
```

The result includes the planner decision, the bounded retrieval-tool trace, the grounded answer, citations, and separate planner and answer token usage.

The API exposes the same workflow at `POST /agent/ask`:

```bash
curl -X POST http://127.0.0.1:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How are the API gateway and autoscaler related?"}'
```

See `docs/agent-orchestration.md` for the routing and failure-handling design.

## Trace agent executions with MLflow

Install the optional observability dependency:

```bash
python -m pip install -e ".[dev,ops]"
```

Start a local MLflow server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

In another terminal, enable tracing for one agent process:

```bash
EKA_MLFLOW_ENABLED=true python -m enterprise_knowledge_agent.agent_query \
  "How are the API gateway and autoscaler related?"
```

Open `http://127.0.0.1:5000` to inspect the resulting trace. The local MLflow database and artifacts are excluded from Git. See `docs/llm-observability.md` for the span model and configuration.

## Extract enterprise entities

Install the optional local NLP dependencies:

```bash
python -m pip install -e ".[dev,nlp]"
```

Run local entity extraction over the prepared enterprise documents:

```bash
python -m enterprise_knowledge_agent.entity_extraction
```

Document-level mentions, canonical entities, and extraction statistics are written under `data/processed/enterprise_rag_bench/entities/`. The full generated records remain excluded from Git, while the small extraction summary is copied to `artifacts/nlp/`.

## Build the knowledge graph

Install the graph dependency and start the local Neo4j service:

```bash
python -m pip install -e ".[dev,nlp,graph]"
docker compose up -d neo4j
```

Build the graph from the generated entity records:

```bash
python -m enterprise_knowledge_agent.graph_build
```

Verify the graph, schema objects, and representative entity relationships:

```bash
python -m enterprise_knowledge_agent.graph_verify
```

Small graph build and verification summaries are written to `artifacts/graph/`. The Neo4j database itself is persisted in a local Docker volume and is not committed to Git.

## Evaluate graph-assisted retrieval

Keep Qdrant and Neo4j running, then inspect one graph-assisted retrieval trace:

```bash
python -m enterprise_knowledge_agent.graph_retrieval "What caused the API gateway incident?"
```

Run the same benchmark used for the dense baseline and write a direct comparison:

```bash
python -m enterprise_knowledge_agent.graphrag_evaluation
```

The comparison in `artifacts/retrieval/graphrag_comparison.json` reports overall and per-question-type metric deltas plus the additional query latency introduced by graph expansion. `docs/graphrag-context.md` documents how those measurements changed the runtime answer architecture.

## Compare answer and agent quality

Keep Qdrant and Neo4j running, then run a balanced local evaluation of dense RAG,
graph-augmented RAG, and the LangGraph agent:

```bash
python -m enterprise_knowledge_agent.answer_evaluation
```

The evaluation backend is provider-selectable. For a Groq run with an explicit shared context
budget across all compared systems:

```bash
python -m enterprise_knowledge_agent.answer_evaluation --provider groq \
  --context-sources 4 --dense-context-sources 3 --graph-context-sources 1 \
  --max-context-characters 8000
```

Provider/model runs use separate artifact files so results from different LLMs are never mixed.
The harness is resumable, paces provider calls, retries transient failures, and writes privacy-safe
per-question rows plus an aggregate comparison to `artifacts/evaluation/`. Failed pairs remain
eligible for retry, and cross-system comparisons are withheld until the selected run is complete.
It measures answerability, citation/document quality, local semantic answer proxies, latency, token
usage, estimated paid-tier token cost, and agent tool-routing behavior. These internal proxy
metrics are not presented as EnterpriseRAG-Bench leaderboard scores. See
`docs/answer-evaluation.md` for metric definitions and limitations.

## Production-style API runtime

The API keeps liveness separate from dependency readiness:

- `GET /health` confirms that the process is alive without contacting external services.
- `GET /ready` checks runtime configuration, the configured Qdrant collection, and Neo4j
  connectivity without consuming language-model API quota.

Responses include an `X-Request-ID` header. Request logs are structured JSON containing method, path,
status, latency, and request ID without logging question bodies. API failures use a stable JSON error
envelope, and the process-level request timeout is configurable with
`EKA_APP_REQUEST_TIMEOUT_SECONDS`.

Build the application container with:

```bash
docker build -t enterprise-knowledge-agent:local .
```

The image runs as a non-root user and uses a single Uvicorn worker so the local embedding model is not
multiplied across worker processes. Runtime credentials are supplied through environment variables and
are not baked into the image. See `docs/production-runtime.md` for the API and container behavior.

## Run the local service stack

The API, Qdrant, Neo4j, and MLflow can run together on one Docker Compose network. Existing Qdrant and
Neo4j named volumes are reused, so the vector index and graph should be built before starting the full
application stack.

```bash
docker compose up -d --build --wait --wait-timeout 180
bash scripts/smoke_stack.sh
```

The API container uses Compose service addresses for Qdrant, Neo4j, and MLflow while credentials stay
in the local `.env` file. Host ports are bound to `127.0.0.1` for local-only access. The smoke test
checks Qdrant, Neo4j, MLflow, `/health`, and `/ready` without making an LLM API call. See
`docs/local-stack.md` for service wiring, persistence, and shutdown behavior.
