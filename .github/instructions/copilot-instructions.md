# Copilot Agent Onboarding Guide

## Quick facts

- **Project**: Retrieval-Augmented Generation (RAG) backend that talks to a Chroma vector store and Google Gemini. Modular Python code in `src/` with pytest-based tests in `tests/`.
- **Languages & tooling**: Python ≥3.11 (repo pins 3.13), dependency management with [`uv`](https://docs.astral.sh/uv/), testing via `pytest`. Docker is used only for the external Chroma service.
- **Repo size & layout**: Compact (
  <150 tracked files). Core code lives under `src/`, tests in `tests/`, infra config under `docker/`. No JS/TS or compiled assets.

## Architecture & code map

- `src/components/chroma_component.py`: Typed wrapper around `chromadb.HttpClient`; defines `ChromaDocument`, `ChromaQueryMatch`, CRUD helpers, and query logic. Raises project-specific exceptions from `src/utils/exceptions.py`.
- `src/components/gemini_embedding.py`: `GeminiEmbeddingService` encapsulates Google Gemini embedding calls with retry/back-off logic driven by configuration.
- `src/config/settings.py`: Pydantic Settings objects (`GeminiSettings`, `ChromaSettings`) hydrate from `.env`, provide helpers like `resolved_timeout()` and `base_url()`.
- `src/utils/exceptions.py`: Centralised runtime exception types.
- `main.py`: Placeholder entry point that currently prints a greeting.
- `docker/docker-compose.yml`: Spins up a Chroma v2 server at `localhost:8000` with persisted volume under `docker/data/`.
- Tests:
  - `tests/test_chroma_component.py`: Unit tests using mocks for the Chroma client.
  - `tests/test_gemini_embedding.py`: Unit tests with patched Gemini SDK.
  - `tests/test_chroma_integration.py`: Live Chroma integration tests (guarded by `@pytest.mark.integration`).
- Config files: `pyproject.toml` (dependencies, pytest config), `.env.example` (baseline env vars), `.python-version` (3.13 runtime hint), `uv.lock` (resolved dependency graph).

## Bootstrap & environment

1. Ensure Python 3.13 is available (validated with `.python-version`). Other 3.11+ runtimes work but stay consistent with CI/runtime expectations.
2. Install [`uv`](https://docs.astral.sh/uv/getting-started/) if absent. Current environment verified with `uv 0.8.19`.
3. From repo root, always run:
   - `uv sync` — installs core dependencies listed in `[project.dependencies]`.
   - `uv sync --group test` — **required** to pull in pytest; `uv run pytest` fails with `program not found` otherwise (observed and confirmed).
4. Copy `.env.example` → `.env` and adjust values as needed. Unit tests monkeypatch Gemini credentials; real runs need a valid `GOOGLE_API_KEY` plus Chroma host info.

## Build, run & validation commands

| Purpose                  | Command (run from repo root)                        | Notes                                                                                                                                                                                                                                                         |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unit + integration tests | `uv run pytest`                                     | After syncing the `test` group, all 17 tests passed in ~5.5s on Windows 11 / Python 3.13.7. Without a live Chroma service the three integration tests will call `pytest.skip`, so failures here usually mean the Docker service is reachable but misbehaving. |
| Unit tests only          | `uv run pytest -m "not integration"`                | Safe default when Chroma isn’t running.                                                                                                                                                                                                                       |
| Start Chroma DB          | `docker compose -f docker/docker-compose.yml up -d` | Requires Docker Desktop. Creates/uses `docker/data/` for persistence. Stop with `docker compose -f docker/docker-compose.yml down`.                                                                                                                           |
| Smoke script             | `uv run python main.py`                             | Currently prints `"Hello from chroma-rag-solution!"`; modify to exercise new features.                                                                                                                                                                        |

There is no dedicated lint or format command yet. If you add one (e.g., Ruff, Black), place config under `pyproject.toml` to keep tooling centralised.

### Integration test behaviour

- `tests/test_chroma_integration.py` will attempt to connect to `http://localhost:8000`. If the Chroma service is down, the fixture catches the exception and triggers `pytest.skip`, so the suite still succeeds.
- When the Docker container is running, the tests perform real CRUD operations and leave the collection empty thanks to the fixture’s cleanup.
- Use the provided volume data in `docker/data/` only as a local cache; CI should provision a clean mount.

## Development workflow tips

- Settings caches (`get_gemini_settings`, `get_chroma_settings`) are memoised with `lru_cache`. Clear them with `.cache_clear()` in tests when overriding env vars (pattern already used in fixtures).
- `GeminiEmbeddingService` performs retries with exponential back-off. When altering retry behaviour, update both the implementation and unit tests that assert the configuration contract.
- `ChromaComponent` methods expect embeddings for upserts; tests enforce this. Maintain the current exceptions for missing embeddings or failed operations to keep error handling consistent.
- Prefer injecting clients/services via constructor args in tests; both components accept optional overrides to ease mocking.

## File inventory snapshot

- Repo root: `.env`, `.env.example`, `.python-version`, `main.py`, `pyproject.toml`, `README.md`, `uv.lock`, `docker/`, `src/`, `tests/`.
- `docker/`: `docker-compose.yml`, `data/` with persisted SQLite + HNSW artifacts (can be pruned for clean states).
- `src/`: `components/`, `config/`, `utils/`, `__init__.py`.
- `tests/`: `conftest.py`, `test_chroma_component.py`, `test_chroma_integration.py`, `test_gemini_embedding.py`.

## Working with CI & verification

- No GitHub Actions are defined yet (`.github/workflows/` absent). Assume reviewers run `uv run pytest` at minimum; add workflows under `.github/workflows/` if you need automated gates.
- When adding features, extend the existing pytest suites and run the full test command locally before committing.
- If contributing integration features, document any new environment variables in `.env.example` and keep `docker/docker-compose.yml` aligned.

## Final guidance

Rely on this document for commands, entry points, and file locations. Only fall back to ad-hoc searches if you encounter a case that contradicts or extends the instructions above. Keeping to these steps should minimise failed builds and review iterations. Happy shipping! 🎯
