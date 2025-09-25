# Chroma RAG Solution

Python-basierte Retrieval-Augmented-Generation (RAG) Plattform: Dokumente werden in einer lokalen Chroma-Instanz gespeichert, Abfragen werden über Google Gemini eingebettet und mit Kontext beantwortet. Eine FastAPI-Anwendung (`POST /chat/query`) stellt die Chat-Funktion nach außen bereit.

## Inhaltsverzeichnis

1. [Architekturüberblick](#architekturüberblick)
2. [Schnellstart](#schnellstart)
3. [API verwenden](#api-verwenden)
4. [Tests & Qualitätssicherung](#tests--qualitätssicherung)
5. [Projektstruktur](#projektstruktur)
6. [Umgebungsvariablen](#umgebungsvariablen)
7. [Weiterführendes](#weiterführendes)

## Architekturüberblick

- `src/components/chroma_component.py` kapselt CRUD- und Query-Operationen gegen Chroma (`chromadb.HttpClient`).
- `src/components/gemini_embedding.py` verwaltet Embedding-Aufrufe an Google Gemini inkl. Retry/Backoff.
- `src/components/gemini_chat.py` orchestriert den RAG-Flow (Embeddings → Chroma → Prompt → Gemini Antwort).
- `src/api/app.py` + `src/api/routes.py` exponieren die Chat-Funktion über FastAPI.
- `src/models/chat.py` definiert Pydantic-Datenmodelle für Requests/Responses.
- `docs/chat_api_design.md` dokumentiert Architekturentscheidungen der Chat-API.

## Schnellstart

### Voraussetzungen

- Python ≥ 3.13 (siehe `.python-version`).
- [uv](https://docs.astral.sh/uv/) als Paket- und Env-Manager (`uv 0.8.19` getestet).
- Docker Desktop (für die optionale lokale Chroma-Instanz).

### Setup

```powershell
cd c:\code\chroma-rag-solution
uv sync
uv sync --group test  # pytest/httpx installieren
copy .env.example .env  # und Werte anpassen
```

### Chroma starten (optional für Integrationstests und Live-Abfragen)

```powershell
docker compose -f docker/docker-compose.yml up -d
```

### API lokal starten

```powershell
uv run python main.py
```

Standardmäßig lauscht die API auf `http://0.0.0.0:8080`. Über Umgebungsvariablen können Host/Port geändert werden (`API_HOST`, `API_PORT`, `API_RELOAD`, `API_LOG_LEVEL`).

### API-Dokumentation (Swagger UI)

Nach dem Start steht unter `http://localhost:8080/docs` eine interaktive Swagger-Oberfläche bereit. Sie zeigt das OpenAPI-Schema, beschreibt alle Parameter und erlaubt Testaufrufe direkt aus dem Browser. Alternativ ist die Spezifikation als JSON unter `http://localhost:8080/openapi.json` abrufbar.

## API verwenden

Endpoint: `POST /chat/query`

```json
{
  "query": "Explain retrieval augmented generation",
  "metadata_filters": { "scope": "public" },
  "override": {
    "temperature": 0.5,
    "top_k": 3,
    "max_output_tokens": 512
  }
}
```

Beispielaufruf via `curl`:

```powershell
curl -X POST http://localhost:8080/chat/query `
     -H "Content-Type: application/json" `
     -d '{"query":"Explain RAG"}'
```

Antwort (`ChatQueryResponse`):

```json
{
  "conversation_id": "...",
  "answer": "Here is the grounded answer...",
  "sources": [
    {
      "id": "doc-123",
      "text": "Snippet",
      "metadata": { "scope": "public" },
      "distance": 0.42
    }
  ],
  "usage": {
    "embedding_ms": 10.0,
    "retrieval_ms": 5.0,
    "generation_ms": 100.0,
    "total_ms": 115.0
  },
  "request_id": "..."
}
```

**Hinweis:** Für produktiven Betrieb muss ein gültiger `GOOGLE_API_KEY` vorhanden sein. Integrationstests überspringen sich automatisch, wenn Chroma nicht erreichbar ist.

## Tests & Qualitätssicherung

- Vollständige Suite: `uv run pytest`
- Nur Unit-Tests (ohne Chroma): `uv run pytest -m "not integration"`
- API-spezifische Tests: `uv run pytest tests/test_chat_api.py`

Alle neuen Features sollten die bestehende Test-Suite erweitern und lokal ausgeführt werden. GitHub Actions sind noch nicht konfiguriert.

## Projektstruktur

```
├── main.py                    # Startet Uvicorn mit FastAPI app
├── src/
│   ├── api/
│   │   ├── app.py             # FastAPI Instanz + Exception Handler
│   │   ├── dependencies.py    # Dependency Injection (ChatService)
│   │   └── routes.py          # /chat/query Route
│   ├── components/
│   │   ├── chroma_component.py
│   │   ├── gemini_chat.py
│   │   └── gemini_embedding.py
│   ├── config/settings.py     # Pydantic Settings (Gemini, Chroma, Chat)
│   ├── models/chat.py         # Pydantic Modelle für API-Verträge
│   └── utils/exceptions.py    # Projektspezifische Fehlerklassen
├── tests/
│   ├── test_chat_api.py
│   ├── test_chroma_component.py
│   ├── test_chroma_integration.py
│   └── test_gemini_embedding.py
├── docker/docker-compose.yml  # Chroma Server v2
├── docs/chat_api_design.md
├── pyproject.toml             # Abhängigkeiten + pytest config
├── uv.lock
└── .env.example
```

## Umgebungsvariablen

| Variable                                              | Beschreibung                                                      |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| `GOOGLE_API_KEY`                                      | API-Key für Google Gemini (Pflicht für echte Anfragen).           |
| `EMBEDDING_MODEL`                                     | Modellname für Embeddings (Standard: `text-embedding-004`).       |
| `CHROMA_HOST` / `CHROMA_PORT`                         | Zieladresse des Chroma-Dienstes.                                  |
| `CHROMA_COLLECTION_NAME`                              | Collection, in die Dokumente geschrieben/abgefragt werden.        |
| `CHROMA_DEFAULT_METADATA`                             | Default-Metadaten als JSON-Objekt.                                |
| `CHAT_MODEL` / `CHAT_MAX_TOKENS` / `CHAT_TEMPERATURE` | Standardparameter für die Chat-Generierung.                       |
| `CHAT_MAX_CONTEXT_DOCS`                               | Anzahl Dokumente, die pro Anfrage aus Chroma geholt werden.       |
| `CHAT_ALLOWED_METADATA_KEYS`                          | Komma-separierte Liste erlaubter Filter (z. B. `scope,category`). |
| `CHAT_DEFAULT_SYSTEM_PROMPT_PATH`                     | Pfad zu einer benutzerdefinierten System-Prompt-Datei.            |
| `API_HOST` / `API_PORT`                               | Bind-Adresse der FastAPI-App (`main.py`).                         |
| `API_RELOAD`                                          | `true`, um Automatisches Reloading zu aktivieren (nur lokal).     |

Weitere Details befinden sich in `.env.example` sowie `src/config/settings.py`.

## Weiterführendes

- `docs/chat_api_design.md` – enthält Design- und Entscheidungsgrundlagen für die API.
- Roadmap-Ideen: Streaming-Responses, Session-Verwaltung, Authentifizierung, strukturierte Logs.

Bei Fragen oder Erweiterungen: Tests schreiben (`uv run pytest`), API manuell testen und Ergebnisse hier dokumentieren.
