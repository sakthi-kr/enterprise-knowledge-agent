# Enterprise Knowledge Agent

A Python project for building and evaluating an enterprise knowledge assistant that combines vector retrieval, graph retrieval, and tool-using LLM workflows over multi-source enterprise data.

## Current implementation

The repository contains a FastAPI application skeleton, environment-based configuration, automated quality checks, and a reproducible data-preparation pipeline for a local EnterpriseRAG-Bench subset. The pipeline downloads a focused engineering corpus, reads source archives directly, preserves document provenance, creates deterministic overlapping chunks, and derives a benchmark question set that is valid for the local corpus.

Retrieval, knowledge-graph construction, agent orchestration, and model-based evaluation are not implemented yet.

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

Generated files are written to `data/processed/enterprise_rag_bench/` and are excluded from Git.
