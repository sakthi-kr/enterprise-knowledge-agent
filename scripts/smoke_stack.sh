#!/usr/bin/env bash
set -euo pipefail

wait_for_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-30}"
  local delay_seconds="${4:-2}"

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --fail --silent --show-error --max-time 5 "$url" >/dev/null; then
      printf '%s: ready\n' "$name"
      return 0
    fi
    sleep "$delay_seconds"
  done

  printf '%s: unavailable after %s attempts\n' "$name" "$attempts" >&2
  return 1
}

docker compose config --quiet
wait_for_http "Qdrant" "http://127.0.0.1:6333/readyz"
wait_for_http "MLflow" "http://127.0.0.1:5000/health"
wait_for_http "API liveness" "http://127.0.0.1:8000/health"
wait_for_http "API readiness" "http://127.0.0.1:8000/ready"

docker compose exec -T neo4j sh -c \
  'cypher-shell -u neo4j -p "${EKA_NEO4J_PASSWORD}" "RETURN 1;" >/dev/null'
printf '%s\n' "Neo4j: ready"
printf '%s\n' "Local stack smoke test passed."
