"""Integration tests against a running Chroma instance."""

from __future__ import annotations

import uuid

import pytest

from src.components.chroma_component import ChromaComponent, ChromaDocument

pytestmark = pytest.mark.integration


@pytest.fixture()
def live_chroma_component():
    """Provide a ChromaComponent backed by the live Docker container."""

    try:
        component = ChromaComponent()
        component.clear_collection()
    except Exception as exc:  # pragma: no cover - executed only when service unavailable
        pytest.skip(f"Chroma service not available: {exc}")

    yield component

    try:
        component.clear_collection()
    except Exception:  # pragma: no cover
        pass


def _sample_document(prefix: str) -> ChromaDocument:
    doc_id = f"{prefix}-{uuid.uuid4()}"
    return ChromaDocument(
        id=doc_id,
        text=f"Integration Test Document {doc_id}",
        metadata={"scope": prefix},
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


def test_upsert_and_get_document(live_chroma_component: ChromaComponent):
    document = _sample_document("get")

    live_chroma_component.upsert_documents([document])
    fetched = live_chroma_component.get_documents([document.id])

    assert len(fetched) == 1
    assert fetched[0].id == document.id
    assert fetched[0].text == document.text
    assert fetched[0].metadata == document.metadata


def test_query_returns_expected_match(live_chroma_component: ChromaComponent):
    document = _sample_document("query")
    live_chroma_component.upsert_documents([document])

    matches = live_chroma_component.query_similar(
        document.embedding,
        top_k=1,
        include_embeddings=True,
    )

    assert matches, "Expected at least one match from Chroma"
    match = matches[0]
    assert match.id == document.id
    assert match.metadata == document.metadata
    assert match.embedding is not None


def test_delete_documents_removes_entries(live_chroma_component: ChromaComponent):
    document = _sample_document("delete")
    live_chroma_component.upsert_documents([document])

    live_chroma_component.delete_documents([document.id])

    remaining = live_chroma_component.get_documents([document.id])
    assert not remaining
    assert live_chroma_component.count() == 0
