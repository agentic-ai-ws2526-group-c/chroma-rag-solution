"""Application configuration management."""

import json
from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    """Configuration values required for interacting with Google Gemini."""

    api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    embedding_model: str = Field(default="text-embedding-004", alias="EMBEDDING_MODEL")
    request_timeout: float = Field(default=60.0, alias="GEMINI_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, alias="GEMINI_MAX_RETRIES")
    retry_backoff_seconds: float = Field(default=2.0, alias="GEMINI_RETRY_BACKOFF_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def resolved_timeout(self, override: Optional[float] = None) -> float:
        """Return the effective timeout, falling back to the configured default."""

        if override is not None and override > 0:
            return override
        return max(self.request_timeout, 0.0)

    def resolved_max_retries(self, override: Optional[int] = None) -> int:
        """Return the effective retry count, ensuring it is non-negative."""

        if override is not None and override >= 0:
            return override
        return max(self.max_retries, 0)

    def resolved_backoff(self, override: Optional[float] = None) -> float:
        """Return the effective backoff duration in seconds."""

        if override is not None and override >= 0:
            return override
        return max(self.retry_backoff_seconds, 0.0)


@lru_cache(maxsize=1)
def get_gemini_settings() -> GeminiSettings:
    """Cache and return the Gemini configuration settings."""

    return GeminiSettings()


class ChromaSettings(BaseSettings):
    """Configuration values required for interacting with Chroma DB."""

    host: str = Field(default="localhost", alias="CHROMA_HOST")
    port: int = Field(default=8000, alias="CHROMA_PORT")
    ssl: bool = Field(default=False, alias="CHROMA_SSL")
    collection_name: str = Field(default="documents", alias="CHROMA_COLLECTION_NAME")
    tenant: Optional[str] = Field(default=None, alias="CHROMA_TENANT")
    auth_token: Optional[str] = Field(default=None, alias="CHROMA_AUTH_TOKEN")
    metadata: dict[str, Any] = Field(default_factory=dict, alias="CHROMA_DEFAULT_METADATA")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> dict[str, Any]:
        """Allow metadata to be provided as JSON strings or dictionaries."""

        if value in (None, "", {}):
            return {}

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - depends on user input
                raise ValueError("CHROMA_DEFAULT_METADATA must be valid JSON.") from exc

            if not isinstance(parsed, dict):
                raise ValueError("CHROMA_DEFAULT_METADATA JSON must describe an object.")
            return parsed

        raise ValueError("CHROMA_DEFAULT_METADATA must be a dict, JSON object, or empty.")

    def base_url(self) -> str:
        """Return the base URL used to communicate with the Chroma server."""

        scheme = "https" if self.ssl else "http"
        return f"{scheme}://{self.host}:{self.port}"


@lru_cache(maxsize=1)
def get_chroma_settings() -> ChromaSettings:
    """Cache and return the Chroma configuration settings."""

    return ChromaSettings()