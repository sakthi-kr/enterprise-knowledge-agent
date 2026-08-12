# Architecture

The system is a local-first enterprise knowledge assistant with two runtime paths: a grounded RAG endpoint and a bounded LangGraph agent endpoint. Both operate over the same Qdrant vector index and Neo4j knowledge graph.

## Data preparation

```text
EnterpriseRAG-Bench
Confluence / Jira / GitHub records
          |
          v
 normalization + stable IDs
          |
          +---------------------------+
          |                           |
          v                           v
 deterministic chunks           source documents
          |                           |
          v                           v
 BGE-small embeddings            GLiNER2 extraction
          |                           |
          v                           v
       Qdrant                 canonical entities
                                      |
                                      v
                                    Neo4j
```

The normalized corpus preserves benchmark `doc_id` values plus stable physical-record IDs. Chunk identifiers are deterministic and retain source provenance.

Qdrant stores 384-dimensional BGE embeddings for chunk retrieval. Neo4j stores source documents, canonical entities, `MENTIONS` relationships, and document-level `CO_OCCURS_WITH` associations.

## Runtime

```text
Client
  |
FastAPI
  |
  +-----------------------------+
  |                             |
POST /ask                 POST /agent/ask
  |                             |
dense retrieval             LangGraph planner
  |                         /             \
bounded graph context   dense only     dense + graph
  |                         \             /
  +--------------------------+-----------+
                             |
                       context assembly
                             |
                           Gemini
                             |
                  grounded answer + citations
                             |
                  optional MLflow tracing
```

### `/ask`

The grounded RAG service retrieves dense Qdrant candidates first. Neo4j can identify additional documents, but graph-derived evidence must satisfy configured support thresholds and receives a limited number of context slots. Graph evidence cannot reorder the dense retrieval ranking.

### `/agent/ask`

The LangGraph workflow carries explicit state through planning, dense retrieval, optional graph expansion, and synthesis. The planner selects one of two strategies:

- `dense_only`
- `dense_plus_graph`

Dense retrieval is always executed. Graph expansion is conditional. The agent has a hard retrieval-tool budget and falls back to dense evidence if planning or graph access fails.

### LLM providers

The application runtime is currently wired to the Gemini REST adapter. The answer-evaluation harness additionally supports Groq through an OpenAI-compatible REST adapter so dense, graph-context, and agent architectures can be compared under one fixed alternative provider/model.

The adapters share provider-neutral planning and grounded-answer contracts.

## Observability

When enabled, MLflow receives a privacy-safe hierarchy:

```text
enterprise_agent                    AGENT
├── plan                            LLM
├── dense_search                    RETRIEVER
├── graph_expand                    TOOL       # conditional
└── synthesize                      LLM
```

The traces retain model/provider names, timings, token counts, routing outcomes, opaque retrieval IDs, and scores. Raw questions, retrieved text, source paths, planner explanations, and generated answer bodies are excluded.

## Service packaging

Docker Compose runs four local services on one network:

```text
API <----> Qdrant
 |
 +-------> Neo4j
 |
 +-------> MLflow
```

Host ports bind to `127.0.0.1`. The API has separate liveness and dependency-readiness endpoints. Qdrant, Neo4j, and MLflow use persistent named volumes.

The stack is designed for reproducible local development and portfolio demonstration, not as an internet-facing deployment.

## Deliberate boundaries

The project does not currently implement:

- live enterprise connectors;
- authentication or tenant isolation;
- ACL-aware retrieval;
- a relational/SQL tool;
- verified business-semantic graph relationships such as ownership or causation;
- a web user interface;
- a production cloud deployment.

These boundaries are kept explicit so implemented capabilities are not confused with possible extensions.
