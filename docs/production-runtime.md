# Production-style API runtime

The API exposes separate liveness and readiness checks.

- `GET /health` confirms that the FastAPI process is alive. It does not contact Qdrant, Neo4j,
  or an external language-model provider.
- `GET /ready` verifies that runtime configuration is present, Qdrant is reachable and contains
  the configured collection, and Neo4j is reachable. It does not make a language-model API call,
  so readiness probes do not consume provider quota.

If a required dependency is unavailable, `/health` remains `200` while `/ready` returns `503`.

Every HTTP response includes an `X-Request-ID` header. A caller-supplied ID is reused only when it
contains safe characters and is at most 64 characters; otherwise the service generates a new ID.
Application request logs are JSON and contain request metadata such as method, path, status, latency,
and request ID. Request bodies and enterprise questions are not logged.

The API applies a process-level request timeout configured with
`EKA_APP_REQUEST_TIMEOUT_SECONDS`. Provider-specific timeouts remain separately configurable through
settings such as `EKA_GEMINI_TIMEOUT_SECONDS`. A timeout returns a stable `504` error envelope instead
of exposing an internal exception.

Expected API failures use a consistent shape:

```json
{
  "error": {
    "code": "retrieval_unavailable",
    "message": "The retrieval service is unavailable.",
    "request_id": "..."
  }
}
```

Runtime clients are created lazily. FastAPI lifespan shutdown closes Qdrant, Neo4j, and language-model
clients that were created by cached services before clearing those caches.

## Container image

The `Dockerfile` installs only the runtime and Neo4j dependencies, runs as an unprivileged user, and
uses one Uvicorn worker to avoid multiplying the local embedding-model memory footprint. Uvicorn access
logs are disabled because the application emits its own request logs with request IDs and latency.

Build the image with:

```bash
docker build -t enterprise-knowledge-agent:local .
```

The container expects configuration through environment variables. The `.env` file is excluded from
the Docker build context and should not be baked into the image.
