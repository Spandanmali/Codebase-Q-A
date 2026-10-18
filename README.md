# codebase-qna

`codebase-qna` is a local-first Python 3.11 project for asking questions about
a codebase. The initial project layout separates ingestion, processing,
retrieval, generation, graph analysis, evaluation, and application concerns so
each area can be implemented independently.

## Project structure

- `src/ingestion/` - Reads source files and other supported input documents.
- `src/chunking/` - Splits documents into searchable, context-sized chunks.
- `src/indexing/` - Builds and persists local search indexes and embeddings.
- `src/retrieval/` - Finds the most relevant chunks for a question.
- `src/graph/` - Represents code relationships such as imports and references.
- `src/generation/` - Produces answers from retrieved context using a local model.
- `src/evaluation/` - Holds answer-quality and retrieval-quality evaluation code.
- `src/app/` - Application entry points and user-facing orchestration.
- `tests/` - Automated tests for configuration and future components.
- `data/` - Local data, indexes, and generated artifacts; intentionally gitignored.
- `config.py` - Typed project settings loaded from `.env` and environment variables.
- `src/ingestion/clone.py` - Safely clones bounded GitHub repositories for ingestion.

## Setup

Python 3.11 is required. The project has no runtime dependency on paid APIs,
API clients, or hosted services.

```text
make install
```

The install target installs this project and the free development tools used
by the `test` and `lint` targets. Copy `.env.example` to `.env` to customize
settings. Defaults are local model names and local filesystem paths; models
still need to be installed or downloaded separately by whichever local
inference implementation is added later.

## Commands

```text
make test
make lint
```

Run `python config.py` to print the active non-secret configuration.

Repository cloning uses `SOURCE_DIR` as its trusted base and creates a fresh
temporary child directory; the caller's destination argument is never used as
a filesystem path. `CLONE_TIMEOUT_SECONDS` defaults to 120 seconds and
`CLONE_MAX_SIZE_BYTES` defaults to 209715200 (200 MB).

## Check a local Ollama server

The repository includes [`scripts/check_ollama.py`](scripts/check_ollama.py), a
small diagnostic that uses Ollama's local HTTP API. It lists installed models,
sends one non-streaming chat request, creates one embedding, prints each call's
latency, and reports measured RAM, disk, and detectable NVIDIA GPU/VRAM.

```text
python scripts/check_ollama.py
python scripts/check_ollama.py --chat-model qwen2.5-coder:3b
python scripts/check_ollama.py --host http://127.0.0.1:11434 --prompt "Say hello"
```

The default chat model is `qwen2.5-coder:7b`; the default embedding model is
`nomic-embed-text`. Both can be changed with command-line options. The models
must already be installed. If one is missing, the diagnostic prints the exact
`ollama pull <name>` command needed.

Ollama exposes a local REST API (normally on `http://127.0.0.1:11434`):

- `GET /api/tags` lists downloaded models.
- `POST /api/chat` accepts a model and message list. The script sets
  `stream: false` so it can measure one complete response.
- `POST /api/embed` accepts a model and input text and returns an embedding
  vector.

The script reports measurements only; it does not claim that a model will run
well. Start Ollama first (for example, `ollama serve`) if the server is not
running.