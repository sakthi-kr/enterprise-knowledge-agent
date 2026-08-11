# Agent orchestration

The agent uses LangGraph as a small, explicit state machine rather than an open-ended tool loop.

```text
question
   |
planner
   |
dense_search
   |
   |-- dense_only --------------------|
   |                                  |
   `-- dense_plus_graph --> graph_expand
                                      |
                                      v
                                  synthesize
                                      |
                               grounded answer
```

## Planner

Gemini returns a schema-constrained routing decision with one of two strategies:

- `dense_only` for direct factual, lookup, and single-topic questions.
- `dense_plus_graph` for questions about relationships, connected entities, dependencies, project context, or cross-document evidence.

The planner does not rewrite the user question and does not answer it. Retrieval always uses the original question so planner wording cannot silently change the search target.

If the planner call fails, the workflow uses dense retrieval as a safe fallback.

## Retrieval tools

`dense_search` queries the existing Qdrant index and is always executed first.

`graph_expand` is conditional. It uses the existing Neo4j graph to identify graph-supported documents, filters weak or already-seen candidates, and asks Qdrant for the best question-matching chunk from each selected graph document.

A graph-tool error is recorded in the tool trace and the workflow continues with dense evidence rather than failing the entire question.

## Bounded execution

The default tool-call limit is two: one dense search and at most one graph expansion. The workflow has no unrestricted agent loop. This makes latency, cost, and failure behavior predictable while still allowing LLM-directed tool selection.

## Answer grounding

The synthesis node uses the same citation validation and insufficient-evidence behavior as the existing RAG path. Dense evidence remains authoritative, and graph evidence is allowed only as a bounded context supplement.
