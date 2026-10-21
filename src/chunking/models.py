"""Shared models for document chunking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A searchable section of a source document."""

    id: str
    file_path: str
    start_line: int
    end_line: int
    text: str
    language: str | None = None
    chunker: str | None = None
    symbol_name: str | None = None
    symbol_type: str | None = None
    parent_class: str | None = None
    docstring: str | None = None
    context: str | None = None
