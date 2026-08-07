"""API health checks."""

from fastapi.testclient import TestClient

from enterprise_knowledge_agent.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Enterprise Knowledge Agent",
        "environment": "development",
    }
