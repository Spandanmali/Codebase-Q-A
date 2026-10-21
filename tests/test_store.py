from pathlib import Path

import pytest

from chunking.models import Chunk
from indexing.store import VectorStore


def make_chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(
        id=chunk_id,
        file_path="example.py",
        start_line=1,
        end_line=1,
        text=text,
        language="python",
        chunker="fixed",
    )


def make_store(path: Path, commit: str = "commit-1") -> VectorStore:
    return VectorStore(path, "owner/repo", commit, "fixed", "test-model")


def test_duplicate_add_upserts_by_chunk_id(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    chunk = make_chunk("chunk-1", "first")

    store.add_chunks([chunk], [[1.0, 0.0]])
    store.add_chunks([make_chunk("chunk-1", "updated")], [[0.0, 1.0]])

    assert store.count() == 1
    assert store.query([0.0, 1.0], 1)[0]["document"] == "updated"


def test_empty_collection_returns_no_results(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    assert store.query([1.0, 0.0], 5) == []


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_chunks([make_chunk("chunk-1")], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="dimension"):
        store.add_chunks([make_chunk("chunk-2")], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        store.query([1.0, 0.0, 0.0], 1)


def test_commits_use_separate_collections(tmp_path: Path) -> None:
    first = make_store(tmp_path, "commit-1")
    second = make_store(tmp_path, "commit-2")
    first.add_chunks([make_chunk("same-id", "first commit")], [[1.0, 0.0]])

    assert first.count() == 1
    assert second.count() == 0
    assert second.query([1.0, 0.0], 1) == []
