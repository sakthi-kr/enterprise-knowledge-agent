# Architecture

The project is designed as a local-first enterprise knowledge assistant that can later combine vector retrieval, graph retrieval, structured-data tools, and LLM-based reasoning.

## Planned data flow

```text
Client
  |
FastAPI
  |
Agent orchestration
  |--------------------|----------------------|
Vector retrieval    Graph retrieval     Structured lookup
  |                    |                      |
Qdrant              Neo4j                 SQL store
  |                    |                      |
  |--------------------|----------------------|
                       |
                 Context assembly
                       |
                 LLM provider
                       |
             Answer with citations
                       |
                    MLflow
              traces and evaluation
```

## Design principles

- Keep the core system runnable on a developer laptop.
- Prefer explicit, testable retrieval and tool interfaces over framework-heavy abstractions.
- Preserve source provenance so generated answers can cite evidence.
- Keep the LLM provider behind a small interface so providers can be changed without rewriting the application.
- Evaluate retrieval and answer quality instead of assuming a more complex architecture is better.
- Add infrastructure only when it supports a working end-to-end system.
