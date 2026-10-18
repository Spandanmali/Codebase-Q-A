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