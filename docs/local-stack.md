# Local service stack

The repository can run the API, Qdrant, Neo4j, and MLflow on one Docker Compose network.
The stack is intended for local development and portfolio demonstration rather than an internet-facing
production deployment.

The host-facing ports are bound to `127.0.0.1` so the local Qdrant, Neo4j, MLflow, and API services
are not exposed to the wider network by default. Container-to-container traffic uses Compose service
names:

- API to Qdrant: `http://qdrant:6333`
- API to Neo4j: `bolt://neo4j:7687`
- API to MLflow: `http://mlflow:5000`

The API container receives provider credentials from `.env`. The file is excluded from Git and from
the Docker build context. Compose overrides only the internal service addresses and enables MLflow
tracing for agent requests.

## Start the stack

Prepare `.env` from `.env.example` and keep the required provider credentials there. The Qdrant
collection and Neo4j graph must already have been built by the repository's indexing workflows. Their
named volumes are reused across Compose restarts.

Start or rebuild the complete stack and wait for health checks:

```bash
docker compose up -d --build --wait --wait-timeout 180
```

The API health check calls `/ready`, so the Compose command does not report the API as healthy until
its configuration is valid, the required Qdrant collection exists, and Neo4j is reachable. MLflow and
Neo4j also have service-level health checks. Qdrant readiness is checked through its `/readyz` endpoint
by the smoke script and indirectly by the API readiness endpoint.

Run the end-to-end infrastructure smoke test from Git Bash:

```bash
bash scripts/smoke_stack.sh
```

The smoke test checks the Compose configuration, Qdrant readiness, MLflow health, API liveness, API
readiness, and a real Neo4j Cypher query. It does not call a language-model provider, so it consumes no
API quota.

Useful local endpoints:

- API: `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`
- MLflow: `http://127.0.0.1:5000`
- Neo4j Browser: `http://127.0.0.1:7474`
- Qdrant: `http://127.0.0.1:6333`

Inspect service state and logs with:

```bash
docker compose ps
docker compose logs api
docker compose logs mlflow
```

Stop the containers without deleting their named volumes:

```bash
docker compose down
```

Do not add `--volumes` unless the local Qdrant index, Neo4j graph, and MLflow state are intentionally
being discarded.

MLflow security middleware is disabled only inside this local Compose setup. Its host port is bound to
loopback, and the service is not intended to be exposed to an untrusted network.
