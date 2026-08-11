# Retrieval artifacts

This directory stores small, reproducible retrieval experiment outputs that are useful for reviewing the project.

The dense-vector baseline writes index statistics, aggregate benchmark metrics, and per-question retrieval results. Graph-assisted retrieval writes its own benchmark metrics and results plus a comparison file showing metric and latency deltas against the dense baseline.

The initial graph-fusion experiment slightly improved rank-10 coverage while reducing early-rank quality and increasing latency. Those results are retained as measured evidence. The runtime answer path uses the graph as a bounded context supplement to dense retrieval instead of treating graph fusion as a universally better ranking method.

Large vector databases, Neo4j storage, and downloaded benchmark data remain local and are excluded from Git.
