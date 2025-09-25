"""Custom exception types used across the project."""


class GeminiConfigurationError(RuntimeError):
    """Raised when the Gemini configuration is invalid or incomplete."""


class GeminiEmbeddingError(RuntimeError):
    """Raised when embedding generation via Gemini fails."""


class ChromaConfigurationError(RuntimeError):
    """Raised when the Chroma configuration is invalid or incomplete."""


class ChromaOperationError(RuntimeError):
    """Raised when a Chroma database operation fails."""


class ChatGenerationError(RuntimeError):
    """Raised when Gemini chat generation fails."""


class ChatValidationError(ValueError):
    """Raised when an incoming chat request fails validation checks."""
