# Graph-assisted retrieval

The graph-assisted retriever starts from dense retrieval rather than asking an LLM to generate Cypher.
This keeps retrieval deterministic and makes the contribution of the knowledge graph measurable.

For each question, the retriever:

1. retrieves a broader dense document candidate set from Qdrant;
2. uses the highest-ranked documents as evidence seeds;
3. finds specific canonical entities mentioned by those documents in Neo4j;
4. expands those entities through document-backed `CO_OCCURS_WITH` relationships;
5. finds other documents mentioning the weighted seed and neighboring entities; and
6. combines dense and graph document rankings with reciprocal-rank fusion.

Very common entities are excluded from graph expansion using their document frequency. Co-occurrence
neighbors also require support from more than one source document by default. These controls reduce the
chance that generic technologies or organization names become graph hubs that dominate retrieval.

The graph score is used only to rank graph candidates. The final ranking uses reciprocal-rank fusion so
that vector similarity and graph evidence remain separate signals rather than being forced onto the same
numeric scale.

This implementation intentionally treats `CO_OCCURS_WITH` as association only. It does not infer causal,
ownership, dependency, or other semantic relationships that are not present in the source evidence.
