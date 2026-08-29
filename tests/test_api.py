from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import legal_ai.api.app as api_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    fake_pipeline = SimpleNamespace(
        explainer=SimpleNamespace(probe=lambda: (True, "test LLM available")),
        retrieval_ready=True,
        retrieval_record_count=10,
        qa_retrieval_ready=True,
        qa_retrieval_record_count=20,
        reference_law_ready=True,
        reference_law_record_count=30,
        retrieval_status_message="",
        qa_retrieval_status_message="",
        reference_law_status_message="",
    )
    monkeypatch.setattr(api_module, "get_pipeline", lambda: fake_pipeline)

    with TestClient(api_module.create_app()) as test_client:
        yield test_client


def test_health_returns_http_200(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_empty_ask_returns_http_422(client: TestClient) -> None:
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
    assert (
        response.json()["detail"][0]["msg"]
        == "Value error, Provide a fuller legal question so retrieval has enough context."
    )
