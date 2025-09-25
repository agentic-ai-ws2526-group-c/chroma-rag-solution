"""API layer tests for the FastAPI chat application."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_chat_service
from src.models import ChatQueryResponse, RetrievedDocument, UsageMetrics
from src.utils.exceptions import ChatGenerationError, ChatValidationError


class StubChatService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = ChatQueryResponse(
            conversation_id=str(uuid4()),
            answer="Here is the grounded answer.",
            sources=[
                RetrievedDocument(
                    id="doc-123",
                    text="Snippet",
                    metadata={"scope": "public"},
                    distance=0.42,
                )
            ],
            usage=UsageMetrics(
                embedding_ms=10.0,
                retrieval_ms=5.0,
                generation_ms=100.0,
                total_ms=115.0,
                total_tokens=256,
            ),
            request_id=str(uuid4()),
        )
        self.raise_error: Exception | None = None

    def generate_response(self, payload):
        self.calls.append({"payload": payload})
        if self.raise_error:
            raise self.raise_error
        return self.response


@pytest.fixture(autouse=True)
def override_chat_service():
    stub = StubChatService()
    app.dependency_overrides[get_chat_service] = lambda: stub
    yield stub
    app.dependency_overrides.clear()


def _client() -> TestClient:
    return TestClient(app)


def test_chat_query_success(override_chat_service):
    client = _client()
    payload = {
        "query": "Tell me about retrieval augmented generation",
        "metadata_filters": {"scope": "public"},
    }

    response = client.post("/chat/query", json=payload)

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["answer"] == "Here is the grounded answer."
    assert body["sources"][0]["id"] == "doc-123"
    assert override_chat_service.calls  # ensure underlying service invoked


def test_chat_query_validation_error(override_chat_service):
    override_chat_service.raise_error = ChatValidationError("Query too short")
    client = _client()

    response = client.post("/chat/query", json={"query": "?"})

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] == "Query too short"


def test_chat_query_generation_error(override_chat_service):
    override_chat_service.raise_error = ChatGenerationError("Gemini timeout")
    client = _client()

    response = client.post("/chat/query", json={"query": "Explain RAG"})

    assert response.status_code == HTTPStatus.BAD_GATEWAY
    assert response.json()["detail"] == "Gemini timeout"
