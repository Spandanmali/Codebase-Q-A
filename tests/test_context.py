from src.chunking.context import build_embedding_text
from src.chunking.models import Chunk


def test_build_embedding_text_adds_metadata_header_without_changing_chunk() -> None:
    chunk = Chunk(
        id="ast:example.py:4-6",
        file_path="example.py",
        start_line=4,
        end_line=6,
        text="    return value\n",
        symbol_name="render",
        symbol_type="method",
        parent_class="Widget",
    )

    embedding_text = build_embedding_text(chunk)

    assert embedding_text == (
        "File: example.py | Parent class: Widget | "
        "Symbol name: render | Symbol type: method\n"
        "    return value\n"
    )
    assert chunk.text == "    return value\n"


def test_build_embedding_text_includes_unset_metadata() -> None:
    chunk = Chunk(
        id="fixed:README.md:1-1",
        file_path="README.md",
        start_line=1,
        end_line=1,
        text="# Project\n",
    )

    assert build_embedding_text(chunk) == (
        "File: README.md | Parent class: None | "
        "Symbol name: None | Symbol type: None\n"
        "# Project\n"
    )
