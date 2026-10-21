from pathlib import Path

import pytest

from src.chunking.ast_chunker import chunk_file, chunk_python
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


def test_chunk_python_preserves_lines_and_structural_metadata() -> None:
    content = '''import os

@decorator
async def top_level():
    \"\"\"top docstring\"\"\"
    def nested():
        return 1
    return nested()

class Outer:
    \"\"\"class docstring\"\"\"
    CONSTANT = 1

    @decorator
    async def method(self):
        return 2

    class Inner:
        VALUE = 3

    def second(self):
        return 4
'''
    chunks = chunk_python("sample.py", content)
    source_lines = content.splitlines()
    covered = [
        line_number
        for chunk in chunks
        for line_number in range(chunk.start_line, chunk.end_line + 1)
        if source_lines[line_number - 1].strip()
    ]
    non_blank = {
        line_number
        for line_number, line in enumerate(source_lines, start=1)
        if line.strip()
    }

    assert set(covered) == non_blank
    assert len(covered) == len(set(covered))
    assert all(chunk.chunker == "ast" for chunk in chunks)
    assert next(chunk for chunk in chunks if chunk.symbol_name == "top_level").start_line == 3
    method = next(chunk for chunk in chunks if chunk.symbol_name == "method")
    assert method.symbol_type == "method"
    assert method.parent_class == "Outer"
    assert method.start_line == 14
    assert method.docstring is None
    outer = next(chunk for chunk in chunks if chunk.symbol_name == "Outer")
    assert outer.end_line == 11
    assert outer.docstring == '"""class docstring"""'
    assert any(chunk.symbol_type == "class_body" for chunk in chunks)
    assert not any(chunk.symbol_name == "nested" for chunk in chunks)


def test_chunk_python_syntax_errors_fall_back_to_fixed(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        chunks = chunk_python("broken.py", "def broken(:\n    pass\n")

    assert chunks
    assert all(chunk.chunker == "fixed" for chunk in chunks)
    assert "Could not parse Python file" in caplog.text


def test_chunk_python_splits_large_symbols_at_statement_boundaries() -> None:
    content = "def large():\n" + "".join(
        f"    value_{number} = {number:030d}\n" for number in range(100)
    )

    chunks = chunk_python("large.py", content)
    large_chunks = [chunk for chunk in chunks if chunk.symbol_name == "large"]

    assert len(large_chunks) > 1
    assert "".join(chunk.text for chunk in large_chunks) == content
    assert all(len(chunk.text) / 4 <= 512 for chunk in large_chunks)
    assert all(chunk.context == "def large():" for chunk in large_chunks)
    assert len({chunk.id for chunk in large_chunks}) == len(large_chunks)
    assert all(
        content.splitlines(keepends=True)[chunk.start_line - 1].startswith("    value_")
        or chunk.start_line == 1
        for chunk in large_chunks
    )


def test_chunk_python_splits_large_statement_by_lines() -> None:
    content = "def large_list():\n    values = [\n" + "".join(
        f"        {number:030d},\n" for number in range(100)
    ) + "    ]\n"

    chunks = chunk_python("large-list.py", content)
    large_chunks = [chunk for chunk in chunks if chunk.symbol_name == "large_list"]

    assert len(large_chunks) > 1
    assert "".join(chunk.text for chunk in large_chunks) == content
    assert [chunk.start_line for chunk in large_chunks] == sorted(
        chunk.start_line for chunk in large_chunks
    )


def test_chunk_file_uses_fixed_chunking_for_documents() -> None:
    chunks = chunk_file("README.md", "# title\n\ntext\n")

    assert len(chunks) == 1
    assert chunks[0].chunker == "fixed"
