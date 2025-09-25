"""Unit tests for the ChatService orchestration layer."""

from __future__ import annotations

from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest

from src.components.chroma_component import ChromaQueryMatch
from src.components.gemini_chat import ChatService
from src.models import ChatParametersOverride, ChatQueryRequest
from src.utils.exceptions import ChatGenerationError, ChatValidationError


def _fake_clock_factory(step: float = 0.05):
    ticks = count()

    def _clock() -> float:
        return next(ticks) * step

    return _clock


class FakeEmbeddingService:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector


class FakeModel:
    def __init__(self, *, text: str, tokens: int | None = None) -> None:
        self.text = text
        self.tokens = tokens
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def generate_content(self, contents, generation_config):
        self.calls.append((contents, generation_config))
        return SimpleNamespace(text=self.text, usage_metadata=SimpleNamespace(total_token_count=self.tokens))


def _default_settings(**overrides):
    base = dict(
        model="gemini-test",
        max_output_tokens=256,
        temperature=0.5,
        max_context_documents=3,
        response_format="text",
        allowed_metadata_keys=["scope"],
        system_prompt_path=None,
        enable_streaming=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_generate_response_happy_path(monkeypatch):
    embedding_service = FakeEmbeddingService([0.1, 0.2, 0.3])

    match = ChromaQueryMatch(
        id="doc-1",
        text="Relevant information about RAG.",
        metadata={"scope": "public"},
        distance=0.12,
        embedding=None,
    )

    chroma = SimpleNamespace()

    def query_similar(embedding, *, top_k, where, include_embeddings):
        chroma.called_with = dict(embedding=embedding, top_k=top_k, where=where, include_embeddings=include_embeddings)
        return [match]

    chroma.query_similar = query_similar

    fake_model = FakeModel(text="Here is the grounded answer.", tokens=321)

    service = ChatService(
        settings=_default_settings(),
        embedding_service=embedding_service,
        chroma_component=chroma,  # type: ignore[arg-type]
        model_factory=lambda _: fake_model,
        system_prompt="You are helpful.",
        clock=_fake_clock_factory(),
    )

    request = ChatQueryRequest(
        query="  What is retrieval augmented generation? ",
        metadata_filters={"scope": "public", "category": "llm"},
        override=ChatParametersOverride(temperature=0.8, top_k=2, max_output_tokens=128, include_embeddings=True),
    )

    response = service.generate_response(request)

    assert response.answer == "Here is the grounded answer."
    assert response.sources and response.sources[0].id == "doc-1"
    assert response.usage.total_tokens == 321
    assert response.conversation_id
    assert response.request_id
    assert response.usage.total_ms > 0

    assert embedding_service.calls == ["What is retrieval augmented generation?"]
    assert chroma.called_with["top_k"] == 2
    assert chroma.called_with["where"] == {"scope": "public"}

    assert fake_model.calls
    contents, generation_config = fake_model.calls[0]
    assert generation_config == {"temperature": 0.8, "max_output_tokens": 128}
    assert contents[0]["role"] == "system"


def test_generate_response_without_matches(monkeypatch):
    embedding_service = FakeEmbeddingService([0.1])

    chroma = SimpleNamespace()
    chroma.query_similar = lambda *_, **__: []

    fake_model = FakeModel(text="I do not have enough information.")

    service = ChatService(
        settings=_default_settings(allowed_metadata_keys=[]),
        embedding_service=embedding_service,
        chroma_component=chroma,  # type: ignore[arg-type]
        model_factory=lambda _: fake_model,
        system_prompt="You are helpful.",
        clock=_fake_clock_factory(),
    )

    request = ChatQueryRequest(query="Explain RAG")
    response = service.generate_response(request)

    assert response.sources == []
    assert "no relevant documents" in fake_model.calls[0][0][1]["parts"][0]


def test_generate_response_empty_query_raises():
    service = ChatService(
        settings=_default_settings(),
        embedding_service=FakeEmbeddingService([0.2]),
        chroma_component=SimpleNamespace(query_similar=lambda *_, **__: []),  # type: ignore[arg-type]
        model_factory=lambda _: FakeModel(text=""),
        system_prompt="You are helpful.",
        clock=_fake_clock_factory(),
    )

    with pytest.raises(ChatValidationError):
        service.generate_response(ChatQueryRequest(query="   "))


def test_generate_answer_failure_raises(monkeypatch):
    embedding_service = FakeEmbeddingService([0.1])

    chroma = SimpleNamespace()
    chroma.query_similar = lambda *_, **__: []

    class FailingModel:
        def generate_content(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    service = ChatService(
        settings=_default_settings(),
        embedding_service=embedding_service,
        chroma_component=chroma,  # type: ignore[arg-type]
        model_factory=lambda _: FailingModel(),
        system_prompt="You are helpful.",
        clock=_fake_clock_factory(),
    )

    with pytest.raises(ChatGenerationError):
        service.generate_response(ChatQueryRequest(query="test"))
