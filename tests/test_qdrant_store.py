"""Tests for the Qdrant REST wrapper."""

import json

import httpx

from enterprise_knowledge_agent.qdrant_store import QdrantStore, QdrantStoreError


def _store_with_handler(handler) -> QdrantStore:
    store = QdrantStore(base_url="http://qdrant.test", timeout_seconds=5)
    store._client.close()
    store._client = httpx.Client(
        base_url="http://qdrant.test",
        transport=httpx.MockTransport(handler),
    )
    return store


def test_qdrant_store_collection_lifecycle_and_count() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/healthz":
            return httpx.Response(200, text='"healthz check passed"')
        if request.method == "GET" and request.url.path == "/collections/demo":
            return httpx.Response(404, json={"status": {"error": "not found"}})
        if request.method == "PUT" and request.url.path == "/collections/demo":
            return httpx.Response(200, json={"result": True, "status": "ok"})
        if request.method == "POST" and request.url.path == "/collections/demo/points/count":
            return httpx.Response(200, json={"result": {"count": 7}, "status": "ok"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with _store_with_handler(handler) as store:
        assert store.health() == "healthz check passed"
        assert store.collection_exists("demo") is False
        store.create_collection(collection_name="demo", vector_size=384)
        assert store.count_points("demo") == 7

    create_request = requests[2]
    assert create_request.read() == b'{"vectors":{"size":384,"distance":"Cosine"}}'


def test_qdrant_store_upsert_query_and_group_query() -> None:
    seen_upsert: dict[str, object] = {}
    seen_query: dict[str, object] = {}
    seen_group_query: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path == "/collections/demo/points":
            assert request.url.params["wait"] == "true"
            seen_upsert.update(json.loads(request.content))
            return httpx.Response(200, json={"result": {"status": "completed"}, "status": "ok"})
        if request.method == "POST" and request.url.path == "/collections/demo/points/query":
            seen_query.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {
                                "id": "chunk-1",
                                "score": 0.91,
                                "payload": {"doc_id": "doc-1"},
                            }
                        ]
                    },
                    "status": "ok",
                },
            )
        if request.method == "POST" and request.url.path == "/collections/demo/points/query/groups":
            seen_group_query.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "result": {
                        "groups": [
                            {
                                "id": "doc-1",
                                "hits": [
                                    {
                                        "id": "chunk-1",
                                        "score": 0.91,
                                        "payload": {"doc_id": "doc-1"},
                                    }
                                ],
                            }
                        ]
                    },
                    "status": "ok",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with _store_with_handler(handler) as store:
        store.upsert_points(
            collection_name="demo",
            points=[{"id": "chunk-1", "vector": [0.1, 0.2], "payload": {"doc_id": "doc-1"}}],
        )
        points = store.query_points(
            collection_name="demo",
            query_vector=[0.2, 0.3],
            limit=5,
        )
        groups = store.query_point_groups(
            collection_name="demo",
            query_vector=[0.2, 0.3],
            group_by="doc_id",
            limit=10,
            group_size=1,
        )

    assert seen_upsert == {
        "points": [{"id": "chunk-1", "vector": [0.1, 0.2], "payload": {"doc_id": "doc-1"}}]
    }
    assert seen_query == {
        "query": [0.2, 0.3],
        "limit": 5,
        "with_payload": True,
        "with_vector": False,
    }
    assert seen_group_query == {
        "query": [0.2, 0.3],
        "group_by": "doc_id",
        "limit": 10,
        "group_size": 1,
        "with_payload": True,
        "with_vector": False,
    }
    assert points[0]["score"] == 0.91
    assert groups[0]["id"] == "doc-1"


def test_qdrant_store_surfaces_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    with _store_with_handler(handler) as store:
        try:
            store.health()
        except QdrantStoreError as exc:
            assert "HTTP 500" in str(exc)
            assert "internal error" in str(exc)
        else:
            raise AssertionError("Expected QdrantStoreError")
