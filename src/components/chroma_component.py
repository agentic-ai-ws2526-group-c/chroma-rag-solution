"""Chroma database component providing a typed interface around the HTTP client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.config.settings import get_chroma_settings
from src.utils.exceptions import ChromaConfigurationError, ChromaOperationError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChromaDocument:
    """Representation of a document stored in Chroma."""

    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[Sequence[float]] = None


@dataclass(slots=True)
class ChromaQueryMatch:
    """A single match returned from a similarity query."""

    id: str
    text: str
    metadata: Optional[Dict[str, Any]]
    distance: float
    embedding: Optional[Sequence[float]]


class ChromaComponent:
    """High-level wrapper around the Chroma client for CRUD and query operations."""

    def __init__(
        self,
        *,
        client: Optional[ClientAPI] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        settings = get_chroma_settings()

        self._client = client or self._build_client(settings)
        target_collection = collection_name or settings.collection_name
        self._collection = self._get_or_create_collection(target_collection, metadata=settings.metadata)

    @property
    def collection(self) -> Collection:
        """Expose the underlying Chroma collection."""

        return self._collection

    def upsert_documents(self, documents: Sequence[ChromaDocument]) -> List[str]:
        """Create or update documents in the active collection and return their IDs."""

        if not documents:
            return []

        missing_embeddings = [doc.id for doc in documents if doc.embedding is None]
        if missing_embeddings:
            raise ValueError(
                "Embeddings are required for all documents. Missing for IDs: " + ", ".join(missing_embeddings)
            )

        try:
            self._collection.upsert(
                ids=[doc.id for doc in documents],
                documents=[doc.text for doc in documents],
                metadatas=[doc.metadata or {} for doc in documents],
                embeddings=[list(doc.embedding or []) for doc in documents],
            )
        except Exception as exc:  # pragma: no cover - error path depends on chromadb implementation
            raise ChromaOperationError("Failed to upsert documents into Chroma.") from exc

        return [doc.id for doc in documents]

    def get_documents(self, ids: Sequence[str]) -> List[ChromaDocument]:
        """Retrieve documents by ID."""

        if not ids:
            return []

        try:
            response = self._collection.get(ids=list(ids), include=["documents", "metadatas", "embeddings"])
        except Exception as exc:  # pragma: no cover
            raise ChromaOperationError("Failed to retrieve documents from Chroma.") from exc

        return self._build_documents_from_response(response)

    def query_similar(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> List[ChromaQueryMatch]:
        """Run a similarity search for the provided embedding."""

        include_fields: List[str] = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include_fields.append("embeddings")

        try:
            result = self._collection.query(
                query_embeddings=[list(embedding)],
                n_results=top_k,
                where=where,
                include=include_fields,
            )
        except Exception as exc:  # pragma: no cover
            raise ChromaOperationError("Failed to execute query against Chroma.") from exc

        return self._build_matches_from_query(result, include_embeddings=include_embeddings)

    def delete_documents(self, ids: Sequence[str]) -> None:
        """Delete documents by ID from the collection."""

        if not ids:
            return

        try:
            self._collection.delete(ids=list(ids))
        except Exception as exc:  # pragma: no cover
            raise ChromaOperationError("Failed to delete documents from Chroma.") from exc

    def clear_collection(self) -> None:
        """Remove all documents from the current collection."""

        try:
            self._collection.delete()
        except Exception as exc:  # pragma: no cover
            raise ChromaOperationError("Failed to clear the Chroma collection.") from exc

    def count(self) -> int:
        """Return the number of items stored in the collection."""

        try:
            stats = self._collection.count()
        except Exception as exc:  # pragma: no cover
            raise ChromaOperationError("Failed to fetch Chroma collection stats.") from exc

        if isinstance(stats, dict):
            return int(stats.get("count", 0))
        return int(stats)

    def _build_client(self, settings) -> ClientAPI:
        headers: Dict[str, str] = {}
        if settings.tenant:
            headers["X-Chroma-Tenant"] = settings.tenant
        if settings.auth_token:
            headers["Authorization"] = f"Bearer {settings.auth_token}"

        try:
            return chromadb.HttpClient(
                host=settings.host,
                port=settings.port,
                ssl=settings.ssl,
                headers=headers or None,
            )
        except Exception as exc:  # pragma: no cover
            raise ChromaConfigurationError("Failed to initialise Chroma client.") from exc

    def _get_or_create_collection(self, name: str, *, metadata: Dict[str, Any]) -> Collection:
        try:
            return self._client.get_or_create_collection(name=name, metadata=metadata or None)
        except Exception as exc:  # pragma: no cover
            raise ChromaOperationError(f"Failed to get or create Chroma collection '{name}'.") from exc

    @staticmethod
    def _build_documents_from_response(response: Dict[str, Any]) -> List[ChromaDocument]:
        ids = response.get("ids", [[]])
        documents = response.get("documents", [[]])
        metadatas = response.get("metadatas", [[]])
        embeddings = response.get("embeddings", [[]])

        if not ids:
            return []

        return [
            ChromaDocument(
                id=item_id,
                text=item_text,
                metadata=item_metadata or None,
                embedding=item_embedding or None,
            )
            for item_id, item_text, item_metadata, item_embedding in zip(
                ids[0], documents[0], metadatas[0], embeddings[0]
            )
        ]

    @staticmethod
    def _build_matches_from_query(
        response: Dict[str, Any], *, include_embeddings: bool
    ) -> List[ChromaQueryMatch]:
        ids = response.get("ids", [[]])
        documents = response.get("documents", [[]])
        metadatas = response.get("metadatas", [[]])
        distances = response.get("distances", [[]])
        embeddings = response.get("embeddings", [[]]) if include_embeddings else [[]]

        if not ids:
            return []

        default_embedding: Sequence[float] | None = None
        if include_embeddings and embeddings:
            default_embedding = []

        matches: List[ChromaQueryMatch] = []
        for index, item_id in enumerate(ids[0]):
            text = documents[0][index] if documents and documents[0] else ""
            metadata = metadatas[0][index] if metadatas and metadatas[0] else None
            distance = float(distances[0][index]) if distances and distances[0] else 0.0
            embedding_value: Optional[Sequence[float]] = None
            if include_embeddings and embeddings and embeddings[0]:
                embedding_value = embeddings[0][index]
            matches.append(
                ChromaQueryMatch(
                    id=item_id,
                    text=text,
                    metadata=metadata,
                    distance=distance,
                    embedding=embedding_value or default_embedding,
                )
            )

        return matches