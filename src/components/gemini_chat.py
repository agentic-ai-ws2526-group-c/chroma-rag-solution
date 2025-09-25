"""Gemini-powered chat orchestration service."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import google.generativeai as genai

from src.components.chroma_component import ChromaComponent, ChromaQueryMatch
from src.components.gemini_embedding import GeminiEmbeddingService
from src.config.settings import ChatSettings, get_chat_settings, get_gemini_settings
from src.models import (
    ChatParametersOverride,
    ChatQueryRequest,
    ChatQueryResponse,
    RetrievedDocument,
    UsageMetrics,
)
from src.utils.exceptions import ChatGenerationError, ChatValidationError

logger = logging.getLogger(__name__)

GenerationModelFactory = Callable[[str], Any]
Clock = Callable[[], float]


class ChatService:
    """Coordinate retrieval augmented chat responses."""

    def __init__(
        self,
        *,
        settings: ChatSettings | None = None,
        embedding_service: GeminiEmbeddingService | None = None,
        chroma_component: ChromaComponent | None = None,
        model_factory: GenerationModelFactory | None = None,
        system_prompt: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings or get_chat_settings()
        self._embedding_service = embedding_service or GeminiEmbeddingService()
        self._chroma_component = chroma_component or ChromaComponent()
        self._model_factory = model_factory or self._default_model_factory
        self._clock = clock or time.perf_counter
        loaded_prompt = system_prompt if system_prompt is not None else self._load_system_prompt(
            self._settings.system_prompt_path
        )
        self._system_prompt = loaded_prompt
        self._system_instruction = self._system_prompt or DEFAULT_SYSTEM_PROMPT

        self._ensure_gemini_configured()

    def generate_response(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """Produce a chat response for the supplied request."""

        normalized_query = request.query.strip()
        if not normalized_query:
            raise ChatValidationError("Query text must not be empty.")

        overrides = request.override or ChatParametersOverride()
        temperature = self._resolve_temperature(overrides.temperature)
        max_output_tokens = self._resolve_max_tokens(overrides.max_output_tokens)
        top_k = self._resolve_top_k(overrides.top_k)
        include_embeddings = bool(overrides.include_embeddings)

        conversation_id = request.conversation_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        timings = _TimingTracker(self._clock)

        embedding = self._embed_query(normalized_query, include_embeddings, timings)
        matches = self._retrieve_context(embedding, top_k, request.metadata_filters, timings)
        prompt = self._build_prompt(normalized_query, matches)
        answer, token_usage = self._generate_answer(prompt, temperature, max_output_tokens, timings)

        usage = UsageMetrics(
            embedding_ms=timings.embedding_ms,
            retrieval_ms=timings.retrieval_ms,
            generation_ms=timings.generation_ms,
            total_ms=timings.total_elapsed_ms,
            total_tokens=token_usage,
        )

        sources = [
            RetrievedDocument(
                id=match.id,
                text=match.text,
                metadata=match.metadata,
                distance=match.distance,
            )
            for match in matches
        ]

        return ChatQueryResponse(
            conversation_id=conversation_id,
            answer=answer,
            sources=sources,
            usage=usage,
            request_id=request_id,
        )

    def _embed_query(
        self,
        query: str,
        include_embeddings: bool,
        timings: "_TimingTracker",
    ) -> list[float]:
        timings.mark("embedding")
        embedding_vector = self._embedding_service.embed_text(query)
        timings.stop("embedding")

        if not embedding_vector:
            raise ChatGenerationError("Gemini returned an empty embedding vector.")

        if include_embeddings:
            logger.debug("Embedding vector length: %s", len(embedding_vector))

        return embedding_vector

    def _retrieve_context(
        self,
        embedding: Iterable[float],
        top_k: int,
        metadata_filters: Optional[dict[str, Any]],
        timings: "_TimingTracker",
    ) -> list[ChromaQueryMatch]:
        where_clause = self._build_metadata_filters(metadata_filters)

        timings.mark("retrieval")
        matches = self._chroma_component.query_similar(
            embedding,
            top_k=top_k,
            where=where_clause,
            include_embeddings=False,
        )
        timings.stop("retrieval")

        return matches

    def _generate_answer(
        self,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        timings: "_TimingTracker",
    ) -> tuple[str, Optional[int]]:
        model = self._model_factory(self._settings.model)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        contents = self._build_contents(prompt)

        timings.mark("generation")
        try:
            response = model.generate_content(
                contents,
                generation_config=generation_config,
            )
        except Exception as exc:  # pragma: no cover - defensive guard around SDK
            raise ChatGenerationError("Gemini text generation failed.") from exc
        finally:
            timings.stop("generation")

        answer = self._extract_text(response)
        token_usage = self._extract_token_usage(response)
        return answer, token_usage

    def _build_prompt(self, query: str, matches: Iterable[ChromaQueryMatch]) -> str:
        sections = []
        for index, match in enumerate(matches, start=1):
            distance_part = f"{match.distance:.4f}" if match.distance is not None else "unknown"
            sections.append(
                f"[Document {index} | distance={distance_part}]\n{match.text.strip()}"
            )

        context_block = "\n\n".join(sections) if sections else "(no relevant documents retrieved)"

        prompt = (
            f"Context:\n{context_block}\n\n"
            f"User question: {query}\n"
            f"Answer:"  # trailing newline handled by Gemini response
        )
        return prompt

    def _build_metadata_filters(self, filters: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not filters:
            return None

        allowed_keys = set(self._settings.allowed_metadata_keys)
        if not allowed_keys:
            logger.debug("Metadata allow list empty; ignoring filters: %s", filters)
            return None

        validated: dict[str, Any] = {}
        for key, value in filters.items():
            if key in allowed_keys:
                validated[key] = value
            else:
                logger.debug("Skipping disallowed metadata filter key: %s", key)

        return validated or None

    def _ensure_gemini_configured(self) -> None:
        settings = get_gemini_settings()
        if not settings.api_key:
            logger.debug("Gemini API key not provided; chat service may fail at runtime.")
            return
        genai.configure(api_key=settings.api_key)

    def _load_system_prompt(self, path: Optional[Path]) -> Optional[str]:
        if not path:
            return None

        expanded = path.expanduser()
        if expanded.is_dir():
            logger.warning("System prompt path is a directory, ignoring: %s", expanded)
            return None
        try:
            content = expanded.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("System prompt file not found: %s", expanded)
            return None
        except OSError as exc:
            logger.warning("Unable to read system prompt file %s: %s", expanded, exc)
            return None

        return content or None

    def _resolve_temperature(self, override: Optional[float]) -> float:
        if override is None:
            return self._settings.temperature
        return max(0.0, min(override, 2.0))

    def _resolve_max_tokens(self, override: Optional[int]) -> int:
        if override is None:
            return self._settings.max_output_tokens
        return max(1, min(override, 8192))

    def _resolve_top_k(self, override: Optional[int]) -> int:
        if override is None:
            return max(1, self._settings.max_context_documents)
        return max(1, min(override, 50))

    def _default_model_factory(self, model_name: str):
        kwargs: dict[str, Any] = {}
        if self._system_instruction:
            kwargs["system_instruction"] = self._system_instruction
        return genai.GenerativeModel(model_name, **kwargs)

    def _build_contents(self, prompt: str) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        contents.append({"role": "user", "parts": [prompt]})
        return contents

    @staticmethod
    def _extract_text(response: Any) -> str:
        if response is None:
            raise ChatGenerationError("Gemini returned an empty response.")

        if hasattr(response, "text") and isinstance(response.text, str):
            text = response.text.strip()
            if text:
                return text

        if isinstance(response, dict):
            text = response.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

            candidates = response.get("candidates")
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts")
                        if isinstance(parts, list) and parts:
                            part_text = parts[0]
                            if isinstance(part_text, str) and part_text.strip():
                                return part_text.strip()

        raise ChatGenerationError("Unable to determine Gemini response text.")

    @staticmethod
    def _extract_token_usage(response: Any) -> Optional[int]:
        usage_attr = getattr(response, "usage_metadata", None)
        if usage_attr is not None:
            total_tokens = getattr(usage_attr, "total_token_count", None)
            if isinstance(total_tokens, int):
                return total_tokens

        if isinstance(response, dict):
            usage_dict = response.get("usageMetadata")
            if isinstance(usage_dict, dict):
                total_tokens = usage_dict.get("totalTokenCount")
                if isinstance(total_tokens, int):
                    return total_tokens

        return None


class _TimingTracker:
    """Utility to capture elapsed times for discrete steps."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._marks: dict[str, float] = {}
        self._splits: dict[str, float] = {}
        self._start = clock()

    def mark(self, label: str) -> None:
        self._marks[label] = self._clock()

    def stop(self, label: str) -> None:
        start = self._marks.get(label)
        if start is None:
            return
        self._splits[label] = (self._clock() - start) * 1000

    @property
    def embedding_ms(self) -> float:
        return self._splits.get("embedding", 0.0)

    @property
    def retrieval_ms(self) -> float:
        return self._splits.get("retrieval", 0.0)

    @property
    def generation_ms(self) -> float:
        return self._splits.get("generation", 0.0)

    @property
    def total_elapsed_ms(self) -> float:
        return (self._clock() - self._start) * 1000


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using the provided context. If the context is insufficient, say you do not have enough information."""
