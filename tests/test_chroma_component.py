"""Unit tests for the ChromaComponent wrapper."""

from types import SimpleNamespace
from unittest import mock

import pytest

from src.components.chroma_component import ChromaComponent, ChromaDocument
from src.utils.exceptions import ChromaConfigurationError, ChromaOperationError


@pytest.fixture(autouse=True)
def clear_chroma_settings_cache():
    from src.config import settings as settings_module

    settings_module.get_chroma_settings.cache_clear()
    yield
    settings_module.get_chroma_settings.cache_clear()


def make_settings(**overrides):
    base = dict(
        host="localhost",
        port=8000,
        ssl=False,
        collection_name="documents",
        tenant=None,
        auth_token=None,
        metadata={},
    )
    base.update(overrides)
    return SimpleNamespace(
        **base,
        base_url=lambda: "http://localhost:8000",
    )


def test_initialization_creates_collection(monkeypatch):
    fake_collection = mock.MagicMock()
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    component = ChromaComponent(client=fake_client)

    expected_metadata = settings.metadata or None
    fake_client.get_or_create_collection.assert_called_once_with(
        name=settings.collection_name, metadata=expected_metadata
    )
    assert component.collection is fake_collection


def test_upsert_documents_requires_embeddings(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = mock.MagicMock()

    component = ChromaComponent(client=fake_client)

    docs = [ChromaDocument(id="a", text="hello")]
    with pytest.raises(ValueError):
        component.upsert_documents(docs)


def test_upsert_documents_calls_collection(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    fake_collection = mock.MagicMock()
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    component = ChromaComponent(client=fake_client)

    docs = [ChromaDocument(id="a", text="hello", embedding=[0.1, 0.2])]
    component.upsert_documents(docs)

    fake_collection.upsert.assert_called_once()


def test_get_documents_builds_records(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    fake_collection = mock.MagicMock()
    fake_collection.get.return_value = {
        "ids": [["a"]],
        "documents": [["hello"]],
        "metadatas": [[{"lang": "en"}]],
        "embeddings": [[[0.1, 0.2]]],
    }
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    component = ChromaComponent(client=fake_client)

    records = component.get_documents(["a"])

    assert len(records) == 1
    assert records[0].id == "a"
    assert records[0].text == "hello"
    assert records[0].metadata == {"lang": "en"}
    assert records[0].embedding == [0.1, 0.2]


def test_query_similar_returns_matches(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    fake_collection = mock.MagicMock()
    fake_collection.query.return_value = {
        "ids": [["a"]],
        "documents": [["hello"]],
        "metadatas": [[{"lang": "en"}]],
        "distances": [[0.12]],
        "embeddings": [[[0.1, 0.2]]],
    }
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    component = ChromaComponent(client=fake_client)

    matches = component.query_similar([0.1, 0.2], top_k=1, include_embeddings=True)

    assert len(matches) == 1
    assert matches[0].id == "a"
    assert matches[0].distance == pytest.approx(0.12)
    assert matches[0].embedding == [0.1, 0.2]


def test_delete_documents_wraps_errors(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    fake_collection = mock.MagicMock()
    fake_collection.delete.side_effect = RuntimeError("boom")
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    component = ChromaComponent(client=fake_client)

    with pytest.raises(ChromaOperationError):
        component.delete_documents(["a"])


def test_clear_collection(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    initial_collection = mock.MagicMock()
    refreshed_collection = mock.MagicMock()
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.side_effect = [initial_collection, refreshed_collection]

    component = ChromaComponent(client=fake_client)
    component.clear_collection()

    fake_client.delete_collection.assert_called_once_with(settings.collection_name)
    assert fake_client.get_or_create_collection.call_count == 2
    assert component.collection is refreshed_collection


def test_count_handles_dict_response(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", lambda: settings)

    fake_collection = mock.MagicMock()
    fake_collection.count.return_value = {"count": 42}
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    component = ChromaComponent(client=fake_client)

    assert component.count() == 42


def test_build_client_raises_configuration_error(monkeypatch):
    settings = make_settings()

    def fake_settings():
        return settings

    monkeypatch.setattr("src.components.chroma_component.get_chroma_settings", fake_settings)

    with mock.patch("chromadb.HttpClient", side_effect=RuntimeError("fail")):
        with pytest.raises(ChromaConfigurationError):
            ChromaComponent(client=None)
