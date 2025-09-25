"""Custom exception types used across the project."""


class GeminiConfigurationError(RuntimeError):
    """Raised when the Gemini configuration is invalid or incomplete."""


class GeminiEmbeddingError(RuntimeError):
    """Raised when embedding generation via Gemini fails."""
