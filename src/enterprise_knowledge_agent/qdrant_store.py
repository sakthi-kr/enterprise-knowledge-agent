"""Minimal Qdrant REST client used by the retrieval baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx


class QdrantStoreError(RuntimeError):
    """Raised when Qdrant returns an unexpected response."""


class QdrantStore:
    """Small REST wrapper for the Qdrant operations this project needs."""

    def __init__(self, *, base_url: str, timeout_seconds: float = 120.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

    def health(self) -> str:
        """Check that the Qdrant service is reachable."""

        response = self._request("GET", "/healthz")
        return response.text.strip().strip('"')

    def collection_exists(self, collection_name: str) -> bool:
        """Return whether a collection exists."""

        response = self._client.get(f"/collections/{collection_name}")
        if response.status_code == 404:
            return False
        self._raise_for_response(response)
        return True

    def create_collection(
        self,
        *,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Create a dense-vector collection."""

        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")
        self._request(
            "PUT",
            f"/collections/{collection_name}",
            json={"vectors": {"size": vector_size, "distance": distance}},
        )

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists."""

        response = self._client.delete(f"/collections/{collection_name}")
        if response.status_code == 404:
            return
        self._raise_for_response(response)

    def upsert_points(self, *, collection_name: str, points: Sequence[dict[str, Any]]) -> None:
        """Insert or update a batch of vector points and wait for completion."""

        if not points:
            return
        self._request(
            "PUT",
            f"/collections/{collection_name}/points",
            params={"wait": "true"},
            json={"points": list(points)},
        )

    def count_points(self, collection_name: str) -> int:
        """Return an exact point count for a collection."""

        response = self._request(
            "POST",
            f"/collections/{collection_name}/points/count",
            json={"exact": True},
        )
        payload = response.json()
        try:
            return int(payload["result"]["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QdrantStoreError("Qdrant count response did not contain result.count") from exc

    def query_points(
        self,
        *,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int,
        query_filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest dense-vector points with payloads."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        request: dict[str, Any] = {
            "query": list(query_vector),
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if query_filter is not None:
            request["filter"] = dict(query_filter)
        response = self._request(
            "POST",
            f"/collections/{collection_name}/points/query",
            json=request,
        )
        payload = response.json()
        try:
            points = payload["result"]["points"]
        except (KeyError, TypeError) as exc:
            raise QdrantStoreError("Qdrant query response did not contain result.points") from exc
        if not isinstance(points, list):
            raise QdrantStoreError("Qdrant query result.points was not a list")
        return points

    def query_point_groups(
        self,
        *,
        collection_name: str,
        query_vector: Sequence[float],
        group_by: str,
        limit: int,
        group_size: int = 1,
        query_filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest vector results grouped by a payload field."""

        if not group_by.strip():
            raise ValueError("group_by must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if group_size <= 0:
            raise ValueError("group_size must be greater than zero")
        request: dict[str, Any] = {
            "query": list(query_vector),
            "group_by": group_by,
            "limit": limit,
            "group_size": group_size,
            "with_payload": True,
            "with_vector": False,
        }
        if query_filter is not None:
            request["filter"] = dict(query_filter)
        response = self._request(
            "POST",
            f"/collections/{collection_name}/points/query/groups",
            json=request,
        )
        payload = response.json()
        try:
            groups = payload["result"]["groups"]
        except (KeyError, TypeError) as exc:
            message = "Qdrant group query response did not contain result.groups"
            raise QdrantStoreError(message) from exc
        if not isinstance(groups, list):
            raise QdrantStoreError("Qdrant group query result.groups was not a list")
        return groups

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise QdrantStoreError(f"Could not reach Qdrant at {self._client.base_url}") from exc
        self._raise_for_response(response)
        return response

    @staticmethod
    def _raise_for_response(response: httpx.Response) -> None:
        if response.is_success:
            return
        body = response.text.strip()
        raise QdrantStoreError(
            f"Qdrant request failed with HTTP {response.status_code}: {body or '<empty body>'}"
        )
