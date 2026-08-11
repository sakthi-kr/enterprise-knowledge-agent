# Retrieval artifacts

This directory stores small, reproducible retrieval experiment outputs that are useful for reviewing the project.

The dense-vector baseline writes index statistics, aggregate benchmark metrics, and per-question retrieval results. Graph-assisted retrieval writes its own benchmark metrics and results plus a comparison file showing metric and latency deltas against the dense baseline.

Large vector databases, Neo4j storage, and downloaded benchmark data remain local and are excluded from Git.
