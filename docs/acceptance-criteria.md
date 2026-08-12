# Completion Criteria

The implementation is considered complete when the core capabilities below are present and the final clean-checkout audit passes.

## Implemented

- [x] Ingest the selected Confluence-, Jira-, and GitHub-style corpus into a normalized document model with stable identifiers and source provenance.
- [x] Build a persistent Qdrant vector index from deterministic chunks.
- [x] Evaluate dense retrieval against benchmark evidence documents.
- [x] Produce grounded answers with validated citations and an explicit insufficient-evidence outcome.
- [x] Extract typed enterprise entities locally and normalize them into stable canonical IDs.
- [x] Build and verify a Neo4j graph using evidence-backed mentions and co-occurrence relationships.
- [x] Compare graph-assisted retrieval against the dense baseline on the same retrieval evaluation set.
- [x] Keep graph-derived answer context bounded after the graph-ranking experiment showed an early-rank regression.
- [x] Run a LangGraph workflow with explicit state, conditional dense/graph tool routing, bounded tool calls, and fallbacks.
- [x] Trace planner, retrieval, graph-tool, synthesis, latency, and token metadata with privacy-safe MLflow spans.
- [x] Compare dense RAG, graph-context RAG, and agent execution under one fixed provider/model and suppress comparisons when a run is incomplete.
- [x] Expose `/health`, `/ready`, `/ask`, and `/agent/ask` through FastAPI with request IDs, structured logs, timeouts, and stable errors.
- [x] Package the API as a non-root container.
- [x] Run the API, Qdrant, Neo4j, and MLflow through one Docker Compose stack with persistent volumes and a smoke test.
- [x] Run linting, formatting checks, and tests in CI on pushes and pull requests.
- [x] Commit measured retrieval, graph, and answer-evaluation summaries without committing raw benchmark content or secrets.
- [x] Document measured regressions and limitations instead of presenting graph or agent complexity as an automatic improvement.

## Final release audit

- [ ] Reproduce the documented quality checks from a clean checkout.
- [ ] Validate documentation links and repository hygiene.
- [ ] Confirm no credentials, local databases, generated corpus data, or machine-specific files are tracked.
- [ ] Validate the Docker/Compose configuration and local smoke-test instructions from the release candidate.
- [ ] Create the final release tag after the audit passes.
