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

Python files are chunked with tree-sitter: the walker visits named children of
the module and class body nodes, keeping decorated definitions together and
recursing into nested classes. Function bodies remain intact so nested
functions stay with their parent. Gaps between structural nodes become module
or class-body chunks. Markdown, TOML, YAML, and JSON use the fixed-size
chunker, and parse failures fall back to it with a warning.

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

Index a GitHub repository with the local embedding and vector-store pipeline:

```text
python -m src.app.cli index https://github.com/owner/repository
python -m src.app.cli index C:\path\to\fixture --local --chunker fixed
```

The local-path option is intended for test fixtures and is available only
through the CLI. Use `--commit` to select a commit, `--force` to reprocess an
already indexed repository, and `--chunker ast` or `--chunker fixed` to choose
the chunking strategy.

Ask a question about all indexed commits for a repository. The `repo_id` must
match the repository identifier used while indexing (a GitHub path such as
`owner/repository`, or the resolved local path):

```text
python -m src.app.cli ask owner/repository "How is authentication handled?"
```

The answer command embeds the question, retrieves the nearest chunks, and
sends only those chunks to the configured local Ollama chat model. It prints
the answer followed by the source file and line ranges used.

Print node and edge counts for a local Python repository graph:

```text
python -m src.app.cli graph-stats C:\path\to\repository
```

The graph builder records files, classes, functions, and routes, with import,
definition, call, and route-handler relationships. Graphs can also be saved
and loaded as JSON through `src.graph.build`.

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

## Embedding dimension

An embedding is a list of numbers that represents text in a mathematical
vector space. Its **dimension** is the number of numbers in that list. For
example, `[0.2, -0.1, 0.8]` has dimension 3. Every vector in an index must have
the same dimension so similarity calculations are valid. The Ollama embedder
learns the dimension from the first response and raises an error if a later
batch returns a different size.