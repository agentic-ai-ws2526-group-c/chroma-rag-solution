"""Gemini embedding service implementation."""

from __future__ import annotations

import logging
import time
from typing import Iterable, List, Sequence

import google.generativeai as genai

from src.config.settings import get_gemini_settings
from src.utils.exceptions import GeminiConfigurationError, GeminiEmbeddingError

logger = logging.getLogger(__name__)


class GeminiEmbeddingService:
    """Encapsulates interactions with Google Gemini embedding endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        embedding_model: str | None = None,
        request_timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        settings = get_gemini_settings()

        self._api_key = api_key or settings.api_key
        self._model = embedding_model or settings.embedding_model
        self._request_timeout = settings.resolved_timeout(request_timeout)
        self._max_retries = settings.resolved_max_retries(max_retries)
        self._retry_backoff_seconds = settings.resolved_backoff(retry_backoff_seconds)

        if not self._api_key:
            raise GeminiConfigurationError(
                "Missing Google Gemini API key. Set GOOGLE_API_KEY in the environment or pass it explicitly."
            )

        if not self._model:
            raise GeminiConfigurationError("Embedding model must be provided for Gemini embeddings.")

        genai.configure(api_key=self._api_key)

    def embed_text(self, text: str, *, request_timeout: float | None = None) -> List[float]:
        """Generate an embedding vector for a single text snippet."""

        if text is None:
            raise ValueError("Text must not be None.")

        normalized = text.strip()
        if not normalized:
            raise ValueError("Text must not be empty or only whitespace.")

        return self._execute_with_retry(lambda: self._embed_single(normalized, request_timeout))

    def embed_documents(
        self, texts: Sequence[str], *, request_timeout: float | None = None
    ) -> List[List[float]]:
        """Generate embeddings for a collection of documents."""

        if texts is None:
            raise ValueError("texts must not be None.")

        embeddings: List[List[float]] = []
        for index, text in enumerate(texts):
            try:
                embeddings.append(self.embed_text(text, request_timeout=request_timeout))
            except Exception as exc:  # pragma: no cover - re-raising to provide index context
                raise GeminiEmbeddingError(f"Failed to embed document at index {index}.") from exc
        return embeddings

    def _execute_with_retry(self, operation) -> List[float]:
        attempts = self._max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except GeminiEmbeddingError:
                raise
            except Exception as exc:  # pragma: no cover - exception coverage determined by tests
                last_error = exc
                logger.warning("Gemini embedding attempt %s/%s failed: %s", attempt, attempts, exc)
                if attempt >= attempts:
                    break
                self._sleep_with_backoff(attempt - 1)

        raise GeminiEmbeddingError("Failed to generate embedding after retries.") from last_error

    def _sleep_with_backoff(self, retry_index: int) -> None:
        delay = self._retry_backoff_seconds * (2**retry_index)
        if delay > 0:
            time.sleep(delay)

    def _embed_single(self, text: str, request_timeout: float | None) -> List[float]:
        timeout = get_gemini_settings().resolved_timeout(request_timeout)
        request_options = {"timeout": timeout} if timeout > 0 else None

        try:
            response = genai.embed_content(
                model=self._model,
                content=text,
                request_options=request_options,
            )
        except Exception as exc:  # pragma: no cover - depends on google client behavior
            raise GeminiEmbeddingError("Gemini embedding request failed.") from exc

        return self._extract_embedding(response)

    @staticmethod
    def _extract_embedding(response) -> List[float]:
        if response is None:
            raise GeminiEmbeddingError("Gemini response was empty.")

        embedding = response.get("embedding") if isinstance(response, dict) else None
        if embedding is None:
            raise GeminiEmbeddingError("Gemini response did not contain an embedding.")

        if isinstance(embedding, dict):
            values = embedding.get("values")
        else:
            values = embedding

        if not isinstance(values, Iterable):  # pragma: no cover - defensive guard
            raise GeminiEmbeddingError("Gemini embedding payload is malformed.")

        vector = [float(value) for value in values]
        if not vector:
            raise GeminiEmbeddingError("Gemini embedding vector is empty.")

        return vector