"""Unit tests for chat-related settings."""

from __future__ import annotations

import importlib

import pytest

from src.config import settings


@pytest.fixture(autouse=True)
def clear_chat_settings_cache():
    settings.get_chat_settings.cache_clear()
    yield
    settings.get_chat_settings.cache_clear()


def test_allowed_metadata_keys_defaults_from_env_file(monkeypatch):
    monkeypatch.delenv("CHAT_ALLOWED_METADATA_KEYS", raising=False)

    cfg = settings.get_chat_settings()

    assert cfg.allowed_metadata_keys == ["scope", "category"]


def test_allowed_metadata_keys_handles_comma_string(monkeypatch):
    monkeypatch.setenv("CHAT_ALLOWED_METADATA_KEYS", "scope, category ,tag")

    cfg = settings.get_chat_settings()

    assert cfg.allowed_metadata_keys == ["scope", "category", "tag"]


def test_allowed_metadata_keys_handles_blank_string(monkeypatch):
    monkeypatch.setenv("CHAT_ALLOWED_METADATA_KEYS", "")

    cfg = settings.get_chat_settings()

    assert cfg.allowed_metadata_keys == []


def test_allowed_metadata_keys_handles_json_array(monkeypatch):
    monkeypatch.setenv("CHAT_ALLOWED_METADATA_KEYS", "[\"scope\", \"category\"]")

    cfg = settings.get_chat_settings()

    assert cfg.allowed_metadata_keys == ["scope", "category"]
