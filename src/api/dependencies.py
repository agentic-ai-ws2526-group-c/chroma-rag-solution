"""Dependency wiring for the FastAPI application."""

from __future__ import annotations

from functools import lru_cache

from src.components.gemini_chat import ChatService


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Provide a singleton ChatService instance for the API layer."""

    return ChatService()
