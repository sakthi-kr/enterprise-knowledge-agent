# Acceptance Criteria

The repository is considered complete when the following capabilities work end to end and are reproducible from a clean checkout:

- A selected enterprise corpus can be ingested into a normalized document model with stable identifiers and source provenance.
- A vector-retrieval baseline can return relevant evidence and produce grounded answers with citations.
- A knowledge graph can be built from extracted entities and relationships and queried for useful cross-document context.
- Graph-assisted retrieval can be compared against the vector baseline on the same evaluation set.
- An agent can choose among retrieval and structured-data tools, use multiple tools when needed, and stop under explicit conditions.
- The system can abstain when available evidence is insufficient.
- LLM calls, retrieval operations, tool calls, latency, and token usage are traceable.
- Evaluation results compare the implemented retrieval strategies using reproducible metrics and a fixed question set.
- The API exposes health and question-answering endpoints with useful error handling.
- Automated tests cover the important data, retrieval, graph, agent, and API behavior.
- The application can be started locally with documented commands.
- Continuous integration runs linting, formatting checks, and tests on every push and pull request.
- The README reports measured results, known limitations, and the exact scope that is implemented.
