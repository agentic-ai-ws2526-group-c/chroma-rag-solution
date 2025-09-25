# Copilot Agent Onboarding Guide

## Quick facts

- **Project**: Retrieval-Augmented Generation backend with a FastAPI surface (`POST /chat/query`) that orchestrates Chroma + Google Gemini.
- **Languages & tooling**: Python ≥3.11 (repo pins 3.13), [`uv`](https://docs.astral.sh/uv/) for dependency/environment management, FastAPI + Uvicorn for HTTP, `pytest` for tests. Docker optional for Chroma.
- **Repo layout**: Compact (<150 files). Key folders: `src/api`, `src/components`, `src/models`, `src/config`, `tests/`, `docker/`, `docs/`.

## Architecture & code map

- `src/api/app.py`: FastAPI instance, registers routes and exception handlers (`ChatValidationError`, `ChatGenerationError`).
- `src/api/routes.py`: Defines `POST /chat/query`, returning `ChatQueryResponse`.
- `src/api/dependencies.py`: Provides a cached `ChatService` via FastAPI dependency injection.
- `src/components/gemini_chat.py`: Core RAG orchestration (embedding → retrieval → prompt → Gemini generation) with timing metrics and metadata filter enforcement.
- `src/components/gemini_embedding.py`: Gemini embedding helper with retry/backoff.
- `src/components/chroma_component.py`: Wrapper around `chromadb.HttpClient` exposing CRUD/query helpers (`ChromaDocument`, `ChromaQueryMatch`).
- `src/models/chat.py`: Pydantic v2 models for requests/responses/overrides.
- `src/config/settings.py`: Settings for Gemini, Chroma, and chat (new `ChatSettings` includes `CHAT_MAX_CONTEXT_DOCS`, `CHAT_ALLOWED_METADATA_KEYS`, etc.).
- `main.py`: Boots Uvicorn pointing at `src.api.app:app`, respects `API_HOST`, `API_PORT`, `API_RELOAD`, `API_LOG_LEVEL`.
- Tests: `tests/test_chat_api.py`, `tests/test_gemini_chat_service.py`, `tests/test_chroma_component.py`, `tests/test_gemini_embedding.py`, `tests/test_chroma_integration.py`.
- Docs: `docs/chat_api_design.md` captures design decisions; keep it updated if behaviour changes.

## Bootstrap & environment

1. Verify Python 3.13 is installed (`py -3.13 --version` on Windows). `.python-version` pins 3.13.
2. Install `uv` if missing (`pip install uv`). Current environment validated with `uv 0.8.19`.
3. From repo root, run:
   - `uv sync`
   - `uv sync --group test` — pulls in `pytest`, `httpx`. Required before running tests.
4. Copy `.env.example` → `.env` and set values. Real Gemini access needs `GOOGLE_API_KEY`. Chroma settings default to `http://localhost:8000`.
5. Start Chroma locally when integration tests or live retrieval is needed: `docker compose -f docker/docker-compose.yml up -d` (stop with `... down`).

## Build, run & validation commands

| Purpose         | Command                                             | Notes                                                                                                                 |
| --------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Full test suite | `uv run pytest`                                     | Runs 24 tests in ~8s on Windows/Python 3.13.7. Integration tests auto-skip if Chroma unavailable.                     |
| Unit tests only | `uv run pytest -m "not integration"`                | Skips Chroma dependency.                                                                                              |
| API server      | `uv run python main.py`                             | Starts Uvicorn. Defaults: host `0.0.0.0`, port `8080`. Respect `API_HOST`, `API_PORT`, `API_RELOAD`, `API_LOG_LEVEL`. |
| Start Chroma DB | `docker compose -f docker/docker-compose.yml up -d` | Mounts `docker/data/` for persistence.                                                                                |

No lint/format tooling yet. If you add one (Ruff/Black/etc.), configure it in `pyproject.toml` and document the command above.

### Integration test behaviour

- `tests/test_chroma_integration.py`: hits the live Chroma service; skips if the client cannot connect.
- `tests/test_chat_api.py`: overrides `get_chat_service` to assert API responses; adjust dependency overrides if you refactor the app.
- `tests/test_gemini_chat_service.py`: uses fake embeddings/models; update expectations if you change prompt formats or timing logic.

## Development workflow tips

- Settings caches are memoised with `lru_cache`. Clear via `.cache_clear()` when tests modify env vars (see fixtures in `tests/`).
- Metadata filter allow list is enforced via `CHAT_ALLOWED_METADATA_KEYS`. Include new keys there and update `.env.example` + tests.
- `ChatService` accepts injected `embedding_service`, `chroma_component`, `model_factory`, and `clock`. Reuse this for deterministic tests.
- Prompt building logs distances (`match.distance`), guarding against `None`. Keep format changes synced with tests.
- `docs/chat_api_design.md` should be updated alongside architectural changes to keep onboarding accurate.

## File inventory snapshot

- Root: `.env`, `.env.example`, `.python-version`, `main.py`, `pyproject.toml`, `README.md`, `uv.lock`, `docker/`, `docs/`, `src/`, `tests/`.
- `docker/`: `docker-compose.yml` + persisted Chroma data.
- `docs/`: `chat_api_design.md` (Phase 1 design outputs).
- `src/`: `api/`, `components/`, `config/`, `models/`, `utils/`.
- `tests/`: `conftest.py` (adds `src/` to `sys.path`) + the test modules listed above.

## Working with CI & verification

- No workflows yet. Assume reviewers expect `uv run pytest` and manual API verification.
- If adding CI: steps should be `uv sync`, `uv sync --group test`, optional Chroma setup, then `uv run pytest`.
- Document any new environment variables in `.env.example`, `README.md`, and this guide.

## Final guidance

Lean on these instructions first; only fall back to manual repo searches when information is missing or inconsistent. Keep changes test-backed (`uv run pytest`), update docs as behaviour shifts, and leave the API runnable via `uv run python main.py`. Happy shipping! 🎯
