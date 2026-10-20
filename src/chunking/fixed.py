"""Fixed-size line chunking."""

from pathlib import Path

from .models import Chunk


def chunk_fixed(
    file_path: str | Path,
    content: str,
    lines: int = 60,
    overlap: int = 10,
) -> list[Chunk]:
    """Split content into fixed-size, overlapping line windows."""
    if lines <= 0:
        raise ValueError("lines must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must not be negative")

    path = str(file_path)
    source_lines = content.splitlines(keepends=True)
    if not source_lines:
        return []

    step = max(lines - min(overlap, lines - 1), 1)
    chunks: list[Chunk] = []
    start = 0

    while start < len(source_lines):
        end = min(start + lines, len(source_lines))
        start_line = start + 1
        end_line = end
        chunks.append(
            Chunk(
                id=f"fixed:{path}:{start_line}-{end_line}",
                file_path=path,
                start_line=start_line,
                end_line=end_line,
                text="".join(source_lines[start:end]),
                chunker="fixed",
            )
        )
        if end == len(source_lines):
            break
        start += step

    return chunks
