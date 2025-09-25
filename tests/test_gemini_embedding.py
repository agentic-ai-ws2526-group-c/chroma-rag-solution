"""Tests for the GeminiEmbeddingService component."""

from unittest import mock

import pytest

from src.components.gemini_embedding import GeminiEmbeddingService
from src.utils.exceptions import GeminiConfigurationError, GeminiEmbeddingError


@pytest.fixture(autouse=True)
def clear_caches():
    """Ensure cached settings are reset between tests."""

    from src.config import settings as settings_module

    settings_module.get_gemini_settings.cache_clear()
    yield
    settings_module.get_gemini_settings.cache_clear()


@mock.patch("google.generativeai.configure")
class TestGeminiEmbeddingService:
    def test_initialization_requires_api_key(self, mock_configure, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        fake_settings = SimpleNamespace(
            api_key="",
            embedding_model="model",
            resolved_timeout=lambda override=None: 60.0,
            resolved_max_retries=lambda override=None: 3,
            resolved_backoff=lambda override=None: 2.0,
        )

        with mock.patch("src.components.gemini_embedding.get_gemini_settings", return_value=fake_settings):
            with pytest.raises(GeminiConfigurationError):
                GeminiEmbeddingService(api_key="")
        mock_configure.assert_not_called()

    def test_embed_text_success(self, mock_configure, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")

        response_payload = {"embedding": [0.1, 0.2, 0.3]}

        with mock.patch("google.generativeai.embed_content", return_value=response_payload) as mock_embed:
            service = GeminiEmbeddingService()
            vector = service.embed_text("Hello world")

        mock_configure.assert_called_once_with(api_key="test-key")
        mock_embed.assert_called_once()
        assert vector == [0.1, 0.2, 0.3]

    def test_embed_text_retries_and_raises(self, mock_configure, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        error = RuntimeError("boom")
        with mock.patch("google.generativeai.embed_content", side_effect=error):
            service = GeminiEmbeddingService(max_retries=1, retry_backoff_seconds=0)
            with pytest.raises(GeminiEmbeddingError):
                service.embed_text("test")

    def test_embed_documents_handles_multiple_entries(self, mock_configure, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        responses = iter([
            {"embedding": [0.4, 0.5]},
            {"embedding": {"values": [0.6, 0.7]}},
        ])

        with mock.patch("google.generativeai.embed_content", side_effect=lambda **_: next(responses)) as mock_embed:
            service = GeminiEmbeddingService()
            vectors = service.embed_documents(["doc1", "doc2"])

        assert vectors == [[0.4, 0.5], [0.6, 0.7]]
        assert mock_embed.call_count == 2

    def test_embed_text_rejects_empty_strings(self, mock_configure, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        service = GeminiEmbeddingService()

        with pytest.raises(ValueError):
            service.embed_text(" ")
