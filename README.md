# Enterprise Knowledge Agent

A Python project for building and evaluating an enterprise knowledge assistant that combines vector retrieval, graph retrieval, and tool-using LLM workflows over multi-source enterprise data.

## Current implementation

The repository contains a FastAPI application, environment-based configuration, automated quality checks, a reproducible EnterpriseRAG-Bench data pipeline, a dense-vector retrieval baseline, grounded answer generation, local enterprise entity extraction, and a Neo4j knowledge graph.

The retrieval baseline uses a local FastEmbed model for CPU inference and Qdrant for persistent vector search, with deterministic chunk provenance and benchmark evaluation against ground-truth document IDs. Grounded answers use a Gemini provider adapter with structured output, validated inline citations, insufficient-evidence handling, and retry logic for transient provider failures.

Entity extraction uses a local GLiNER2 model with descriptive labels to identify stable named enterprise entities such as people, organizations, teams, projects, services, technologies, and repositories. Entity mentions are normalized into stable canonical IDs.

Neo4j stores source documents and canonical entities with evidence-backed `MENTIONS` relationships and document-level `CO_OCCURS_WITH` relationships. The graph keeps association separate from stronger semantic claims such as ownership or causation.

Graph-assisted retrieval, agent orchestration, and model-based evaluation are not implemented yet.

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

Ask a question from the command line:

```bash
python -m enterprise_knowledge_agent.rag_query "What caused the API gateway autoscaler incident?"
```

Or start the API and send a request to `POST /ask`:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What caused the API gateway autoscaler incident?"}'
```

The response includes an answer status, validated citations with source provenance, retrieval counts, model name, and provider token usage.

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
