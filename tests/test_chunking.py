from pathlib import Path

import pytest

from src.chunking.fixed import chunk_fixed
from src.chunking.models import Chunk


def test_chunk_fixed_empty_file_returns_no_chunks(tmp_path: Path) -> None:
    assert chunk_fixed(tmp_path / "empty.py", "") == []


def test_chunk_fixed_shorter_than_window_returns_one_chunk(tmp_path: Path) -> None:
    file_path = tmp_path / "short.py"
    chunks = chunk_fixed(file_path, "one\ntwo\n", lines=60, overlap=10)

    assert chunks == [
        Chunk(
            id=f"fixed:{file_path}:1-2",
            file_path=str(file_path),
            start_line=1,
            end_line=2,
            text="one\ntwo\n",
            language=None,
            chunker="fixed",
            symbol_name=None,
            symbol_type=None,
            parent_class=None,
            docstring=None,
        )
    ]


def test_chunk_fixed_large_overlap_still_makes_progress() -> None:
    content = "".join(f"line {number}\n" for number in range(1, 6))

    chunks = chunk_fixed("example.py", content, lines=3, overlap=3)

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 3),
        (2, 4),
        (3, 5),
    ]


def test_chunk_fixed_ids_are_unique_and_stable() -> None:
    content = "".join(f"line {number}\n" for number in range(1, 7))

    chunks = chunk_fixed("src/example.py", content, lines=3, overlap=1)

    assert [chunk.id for chunk in chunks] == [
        "fixed:src/example.py:1-3",
        "fixed:src/example.py:3-5",
        "fixed:src/example.py:5-6",
    ]
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_chunk_fixed_rejects_invalid_window_values() -> None:
    with pytest.raises(ValueError, match="lines"):
        chunk_fixed("example.py", "content", lines=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_fixed("example.py", "content", overlap=-1)
