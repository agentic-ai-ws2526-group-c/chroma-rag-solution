"""FastAPI application exposing the chat service."""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api import routes
from src.utils.exceptions import ChatGenerationError, ChatValidationError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chroma RAG Chat API",
    version="0.1.0",
    description=(
        "REST surface for the Chroma-backed Retrieval-Augmented Generation service. "
        "Use the Swagger UI at /docs to explore request/response contracts."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.include_router(routes.router)


@app.exception_handler(ChatValidationError)
async def handle_chat_validation_error(request: Request, exc: ChatValidationError):
    logger.info("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=HTTPStatus.BAD_REQUEST.value,
        content={"detail": str(exc)},
    )


@app.exception_handler(ChatGenerationError)
async def handle_chat_generation_error(request: Request, exc: ChatGenerationError):
    logger.exception("Chat generation failure on %s", request.url.path)
    return JSONResponse(
        status_code=HTTPStatus.BAD_GATEWAY.value,
        content={"detail": str(exc)},
    )
