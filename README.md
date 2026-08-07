# Enterprise Knowledge Agent

A Python project for building and evaluating an enterprise knowledge assistant that combines vector retrieval, graph retrieval, and tool-using LLM workflows over multi-source enterprise data.

## Current implementation

The repository currently contains a FastAPI application skeleton, environment-based configuration, an API health check, automated tests, and CI quality checks. Retrieval, knowledge-graph construction, agent orchestration, and evaluation will be added as the system develops.

## Development

Python 3.10 is the target development version.

Create and activate a virtual environment, install the development dependencies, and run the checks:

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
