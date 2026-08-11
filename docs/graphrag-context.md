# Graph-Augmented Answer Context

The first graph-assisted retrieval experiment used reciprocal-rank fusion to combine dense and graph document rankings. On the 177-question retrieval benchmark, graph expansion recovered a small amount of additional evidence at rank 10 but substantially reduced early-rank quality.

| Metric | Dense | Graph fusion | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@1 | 0.468927 | 0.305085 | -0.163842 |
| Hit Rate@10 | 0.661017 | 0.666667 | +0.005650 |
| MRR@10 | 0.533764 | 0.414035 | -0.119729 |
| Recall@10 | 0.603107 | 0.607345 | +0.004238 |
| Mean latency | 53.804 ms | 134.139 ms | +80.335 ms |

The result suggests that the co-occurrence graph is useful as an evidence-expansion mechanism but is too noisy to replace dense relevance ranking. The runtime answer path therefore keeps dense retrieval as the primary ranking signal and uses graph retrieval only to reserve a small amount of additional context.

The default answer context contains up to six evidence chunks:

- the first four slots are reserved for dense retrieval;
- up to two slots can be filled by graph-derived documents;
- graph documents must be outside the dense chunk result set and match at least two selected graph entities;
- if fewer graph documents qualify, unused slots are filled by additional dense chunks;
- every graph-selected document is queried again in Qdrant so the language model receives the chunk from that document that is most relevant to the user's question.

This design preserves the strong early ranking of the dense retriever while still allowing the graph to contribute evidence that vector search did not surface in the answer context. The original graph-fusion benchmark remains in `artifacts/retrieval/` as an experimental result rather than being hidden or overwritten.
