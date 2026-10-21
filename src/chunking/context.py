"""Helpers for preparing chunk context for embedding and display."""

from .models import Chunk


def build_embedding_text(chunk: Chunk) -> str:
    """Return metadata-rich text for embedding while leaving the chunk unchanged."""
    header = (
        f"File: {chunk.file_path} | "
        f"Parent class: {chunk.parent_class} | "
        f"Symbol name: {chunk.symbol_name} | "
        f"Symbol type: {chunk.symbol_type}"
    )
    return f"{header}\n{chunk.text}"
