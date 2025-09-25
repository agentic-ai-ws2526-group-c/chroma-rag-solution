"""Application configuration management."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
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