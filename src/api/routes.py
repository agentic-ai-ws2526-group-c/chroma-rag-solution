"""FastAPI routes for the chat API."""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import APIRouter, Depends

from src.api.dependencies import get_chat_service
from src.components.gemini_chat import ChatService
from src.models import ChatQueryRequest, ChatQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatQueryResponse, status_code=HTTPStatus.OK)
async def query_chat(
    payload: ChatQueryRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatQueryResponse:
    """Execute a RAG-powered chat query."""
    logger.debug("Processing chat request for conversation_id=%s", payload.conversation_id)
    return service.generate_response(payload)
