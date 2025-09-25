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
        self._collection_name = collection_name or settings.collection_name
        self._collection_metadata = settings.metadata
        self._collection = self._get_or_create_collection(
            self._collection_name, metadata=self._collection_metadata
        )

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
            self._client.delete_collection(self._collection_name)
        except Exception as exc:  # pragma: no cover
            raise ChromaOperationError("Failed to clear the Chroma collection.") from exc

        self._collection = self._get_or_create_collection(
            self._collection_name, metadata=self._collection_metadata
        )

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
        doc_ids = ChromaComponent._flatten(response.get("ids", []))
        doc_texts = ChromaComponent._flatten(response.get("documents", []))
        doc_metadatas = ChromaComponent._flatten(response.get("metadatas", []))
        doc_embeddings = ChromaComponent._flatten(response.get("embeddings", []))

        if not doc_ids:
            return []

        records: List[ChromaDocument] = []
        for index, item_id in enumerate(doc_ids):
            text = doc_texts[index] if index < len(doc_texts) else ""
            metadata = doc_metadatas[index] if index < len(doc_metadatas) else None
            embedding = doc_embeddings[index] if index < len(doc_embeddings) else None

            if isinstance(metadata, dict) and not metadata:
                metadata = None

            records.append(
                ChromaDocument(
                    id=str(item_id),
                    text=str(text) if text is not None else "",
                    metadata=metadata,
                    embedding=ChromaComponent._coerce_embedding(embedding),
                )
            )

        return records

    @staticmethod
    def _build_matches_from_query(
        response: Dict[str, Any], *, include_embeddings: bool
    ) -> List[ChromaQueryMatch]:
        ids = ChromaComponent._flatten(response.get("ids", []))
        documents = ChromaComponent._flatten(response.get("documents", []))
        metadatas = ChromaComponent._flatten(response.get("metadatas", []))
        distance_values = ChromaComponent._flatten(response.get("distances", []))
        embedding_values = (
            ChromaComponent._flatten(response.get("embeddings", [])) if include_embeddings else []
        )

        if not ids:
            return []

        matches: List[ChromaQueryMatch] = []
        for index, item_id in enumerate(ids):
            text = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else None
            distance_raw = distance_values[index] if index < len(distance_values) else 0.0
            embedding_value = (
                embedding_values[index] if index < len(embedding_values) else None
            )

            matches.append(
                ChromaQueryMatch(
                    id=str(item_id),
                    text=str(text) if text is not None else "",
                    metadata=metadata,
                    distance=float(distance_raw),
                    embedding=ChromaComponent._coerce_embedding(embedding_value),
                )
            )

        return matches

    @staticmethod
    def _flatten(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return [item for sub in value for item in sub]
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _coerce_embedding(value: Any) -> Optional[Sequence[float]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        try:
            import numpy as np

            if isinstance(value, np.ndarray):  # pragma: no cover - requires numpy at runtime
                return value.astype(float).tolist()
        except Exception:  # pragma: no cover - numpy optional
            pass

        return None