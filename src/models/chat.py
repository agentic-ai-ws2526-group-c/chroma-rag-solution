"""Pydantic models describing chat API contracts."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatParametersOverride(BaseModel):
    """Optional per-request overrides for chat generation behaviour."""

    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, gt=0, le=50)
    max_output_tokens: Optional[int] = Field(default=None, gt=0, le=8192)
    include_embeddings: Optional[bool] = None


class ChatQueryRequest(BaseModel):
    """Inbound payload for a chat query."""

    query: str = Field(min_length=1)
    conversation_id: Optional[str] = None
    metadata_filters: Optional[dict[str, Any]] = None
    override: Optional[ChatParametersOverride] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Explain the benefits of retrieval augmented generation.",
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "metadata_filters": {"scope": "public"},
                "override": {"temperature": 0.4, "top_k": 3},
            }
        }
    }


class RetrievedDocument(BaseModel):
    """Document snippet returned from the vector store."""

    id: str
    text: str
    metadata: Optional[dict[str, Any]] = None
    distance: Optional[float] = None


class UsageMetrics(BaseModel):
    """Timing and token usage metadata for a chat request."""

    embedding_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    total_tokens: Optional[int] = None


class ChatQueryResponse(BaseModel):
    """Outbound payload for chat responses."""

    conversation_id: str
    answer: str
    sources: list[RetrievedDocument]
    usage: UsageMetrics
    request_id: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "answer": "Here is the grounded response...",
                "sources": [
                    {
                        "id": "doc-001",
                        "text": "Relevant excerpt",
                        "metadata": {"scope": "public"},
                        "distance": 0.12,
                    }
                ],
                "usage": {
                    "embedding_ms": 12.3,
                    "retrieval_ms": 4.5,
                    "generation_ms": 180.6,
                    "total_ms": 197.4,
                    "total_tokens": 356,
                },
                "request_id": "791f7844-8b0d-4d32-a3f5-5f6dc0b36521",
            }
        }
    }
