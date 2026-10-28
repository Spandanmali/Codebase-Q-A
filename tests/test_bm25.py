from pathlib import Path

from chunking.models import Chunk
from retrieval.bm25 import BM25Index, tokenize_code


def make_chunk(chunk_id: str, text: str, symbol_name: str | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        file_path="src/auth/token_utils.py",
        start_line=1,
        end_line=1,
        text=text,
        chunker="ast",
        symbol_name=symbol_name,
    )


def test_tokenizer_splits_code_names_and_paths() -> None:
    assert tokenize_code("verifyToken src/auth/token_utils.py") == [
        "verify",
        "token",
        "src",
        "auth",
        "token",
        "utils",
        "py",
    ]


def test_search_matches_camel_case_query_to_snake_case_chunk(tmp_path: Path) -> None:
    index = BM25Index(tmp_path, "owner/repo", "commit-1", "ast")
    index.build(
        [
            make_chunk("matching", "def verify_token():", "verify_token"),
            make_chunk("other", "def render_page():", "render_page"),
        ]
    )

    assert index.search("verifyToken", 1)[0][0] == "matching"
    assert index.path.is_file()
    assert BM25Index(tmp_path, "owner/repo", "commit-1", "ast").load().search(
        "verifyToken", 1
    )[0][0] == "matching"


def test_persisted_index_loads_and_empty_corpus_searches(tmp_path: Path) -> None:
    index = BM25Index(tmp_path, "owner/repo", "commit-1", "fixed")
    index.build([])

    loaded = BM25Index(tmp_path, "owner/repo", "commit-1", "fixed").load()

    assert loaded.search("anything", 5) == []
