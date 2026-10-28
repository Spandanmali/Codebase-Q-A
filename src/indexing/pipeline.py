"""Repository indexing orchestration."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

try:
    from chunking import build_embedding_text, chunk_file
    from chunking.fixed import chunk_fixed
    from ingestion.clone import clone_repo
    from ingestion.discover import discover_files
    from retrieval.bm25 import BM25Index
except ModuleNotFoundError:
    from ..chunking import build_embedding_text, chunk_file
    from ..chunking.fixed import chunk_fixed
    from ..ingestion.clone import clone_repo
    from ..ingestion.discover import discover_files
    from ..retrieval.bm25 import BM25Index
from config import load_settings

from .embedder import Embedder
from .store import VectorStore

LOGGER = logging.getLogger(__name__)


def _repo_name(source: str | Path) -> str:
    if isinstance(source, Path):
        return str(source.resolve())
    parsed = urlsplit(source)
    return parsed.path.removesuffix(".git").strip("/") or source


def _local_commit(path: Path, requested: str | None) -> str:
    if requested:
        return requested
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "local"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "local"


def _log_stage(stage: str, started: float) -> None:
    LOGGER.info("index stage %s completed in %.3fs", stage, time.perf_counter() - started)


def index_repo(
    source: str | Path,
    chunker: str = "ast",
    commit: str | None = None,
    force: bool = False,
    *,
    allow_local: bool = False,
) -> dict[str, int | float]:
    """Clone or read a repository and persist its embedded chunks.

    Local paths are deliberately opt-in for the CLI's fixture mode. API callers
    must provide a GitHub URL.
    """
    if chunker not in {"ast", "fixed"}:
        raise ValueError("chunker must be 'ast' or 'fixed'")

    local_path = Path(source) if isinstance(source, Path) else None
    if local_path is not None and not allow_local:
        raise ValueError("local repository paths are only supported by the CLI --local mode")
    if local_path is not None and not local_path.is_dir():
        raise ValueError(f"local repository path is not a directory: {local_path}")
    if local_path is None and not isinstance(source, str):
        raise TypeError("source must be a GitHub URL or a local Path in CLI mode")

    started_total = time.perf_counter()
    settings = load_settings()
    working_path: Path
    cloned_path: Path | None = None
    repository = _repo_name(source)
    try:
        clone_started = time.perf_counter()
        if local_path is not None:
            working_path = local_path.resolve()
            actual_commit = _local_commit(working_path, commit)
        else:
            cloned_path, actual_commit = clone_repo(source, settings.source_dir, commit)
            working_path = cloned_path
        _log_stage("clone", clone_started)

        store = VectorStore(
            settings.index_dir,
            repository,
            actual_commit,
            chunker,
            settings.embedding_model,
        )
        bm25_index = BM25Index(settings.index_dir, repository, actual_commit, chunker)
        if not force and store.is_indexed():
            LOGGER.info(
                "repository %s at %s with %s chunks is already indexed; skipping",
                repository,
                actual_commit,
                chunker,
            )
            return {
                "files": 0,
                "skipped": 0,
                "chunks": store.count(),
                "time": time.perf_counter() - started_total,
            }

        discover_started = time.perf_counter()
        files = discover_files(working_path)
        _log_stage("discover", discover_started)
        if not files:
            LOGGER.info("repository %s is empty or has no supported files", repository)
            bm25_index.build([])
            store.mark_indexed()
            return {
                "files": 0,
                "skipped": 0,
                "chunks": 0,
                "time": time.perf_counter() - started_total,
            }

        stats: dict[str, int | float] = {"files": 0, "skipped": 0, "chunks": 0}
        indexed_chunks = []
        for entry in files:
            relative_path = str(entry["path"])
            file_path = working_path / relative_path
            try:
                content = file_path.read_text(encoding="utf-8")
                chunk_started = time.perf_counter()
                chunks = (
                    chunk_file(relative_path, content)
                    if chunker == "ast"
                    else chunk_fixed(relative_path, content)
                )
                _log_stage(f"chunk {relative_path}", chunk_started)
                if not chunks:
                    stats["files"] += 1
                    continue

                embed_started = time.perf_counter()
                vectors = Embedder().embed([build_embedding_text(chunk) for chunk in chunks])
                _log_stage(f"embed {relative_path}", embed_started)
                store_started = time.perf_counter()
                store.add_chunks(chunks, vectors)
                indexed_chunks.extend(chunks)
                _log_stage(f"store {relative_path}", store_started)
                stats["files"] += 1
                stats["chunks"] += len(chunks)
            except (OSError, UnicodeError, RuntimeError, ValueError) as error:
                LOGGER.exception("skipping %s: %s", relative_path, error)
                stats["skipped"] += 1

        if stats["files"] == 0 and stats["skipped"] > 0:
            LOGGER.info("no files could be indexed; skipped %s file(s)", stats["skipped"])
        bm25_index.build(indexed_chunks)
        store.mark_indexed()
        stats["time"] = time.perf_counter() - started_total
        return stats
    finally:
        if cloned_path is not None:
            shutil.rmtree(cloned_path, ignore_errors=True)
        LOGGER.info("index pipeline completed in %.3fs", time.perf_counter() - started_total)
