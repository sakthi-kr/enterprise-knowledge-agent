# LLM and agent observability

The agent can emit manual MLflow traces without making tracing a hard runtime dependency. Tracing is disabled by default and is enabled with `EKA_MLFLOW_ENABLED=true` when an MLflow tracking server is available.

## Trace structure

One agent request is recorded as a root `AGENT` span with child spans for the operations that actually execute:

```text
enterprise_agent                    AGENT
├── plan                            LLM
├── dense_search                    RETRIEVER
├── graph_expand                    TOOL       # conditional
└── synthesize                      LLM
```

A dense-only request therefore has no `graph_expand` span. Planner failure still produces a planner span but routes to the existing dense fallback. Graph-tool failure is recorded in the graph span output and the workflow continues with dense evidence.

## Safe trace payloads

Trace payloads are metadata-first. Raw user questions, retrieved chunk text, human-readable source paths, planner explanations, and generated answer text are deliberately excluded from MLflow spans. This keeps observability useful without turning the tracing store into a second copy of enterprise content.

The root and LLM spans record bounded request metadata, execution outcomes, model/provider identifiers, and token usage. Query metadata includes character counts rather than raw text.

The dense retrieval span keeps MLflow's retriever-compatible structure but redacts `page_content`. It records opaque chunk, document, and record identifiers, source type, and similarity score. These identifiers are sufficient for local debugging and benchmark joins while avoiding document bodies and source paths.

The graph span records candidate and selected-document counts rather than the selected identifiers themselves. The synthesis span records answer status and citation identifiers, but not the generated answer body.

Retrieval-aware answer evaluation uses the benchmark artifacts directly rather than depending on raw content stored inside traces.

## Local tracking

The default tracking URI is `http://127.0.0.1:5000` and the default experiment name is `enterprise-knowledge-agent`. They can be overridden with:

```text
EKA_MLFLOW_TRACKING_URI
EKA_MLFLOW_EXPERIMENT_NAME
```

Local MLflow state such as `mlflow.db`, `mlartifacts/`, and `.mlflow/` is excluded from Git.

## Failure behavior

Normal application execution uses a no-op tracer when observability is disabled. This keeps MLflow outages or missing optional dependencies from affecting the default RAG and agent paths. When MLflow tracing is explicitly enabled, configuration or tracking-server failures are surfaced rather than silently discarding observability data.
