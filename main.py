"""Entrypoint for running the FastAPI chat service with uvicorn."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Launch the FastAPI application via uvicorn."""

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    reload_flag = os.getenv("API_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload_flag,
        log_level=os.getenv("API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
