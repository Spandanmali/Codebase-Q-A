"""Code-aware BM25 retrieval for persisted repository chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from rank_bm25 import BM25Okapi

try:
    from chunking.models import Chunk
except ModuleNotFoundError:
    from ..chunking.models import Chunk


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")


def tokenize_code(text: str) -> list[str]:
    """Split code identifiers and paths into lowercase searchable tokens."""
    camel_split = _CAMEL_CASE_BOUNDARY.sub(" ", text)
    return [
        token.lower()
        for token in _TOKEN_SEPARATOR.split(camel_split)
        if token
    ]


class BM25Index:
    """A commit- and chunker-specific BM25 index over source chunks."""

    def __init__(
        self,
        persist_dir: str | Path,
        repo: str,
        commit: str,
        chunker: str,
    ) -> None:
        if not repo.strip():
            raise ValueError("repo must not be empty")
        if not commit.strip():
            raise ValueError("commit must not be empty")
        if not chunker.strip():
            raise ValueError("chunker must not be empty")
        self._persist_dir = Path(persist_dir)
        self._repo = repo
        self._commit = commit
        self._chunker = chunker
        self._path = self._persist_dir / self._filename(repo, commit, chunker)
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _filename(repo: str, commit: str, chunker: str) -> str:
        identity = "\0".join((repo, commit, chunker))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"bm25-{digest}.json"

    @staticmethod
    def _document_tokens(chunk: Chunk) -> list[str]:
        searchable = " ".join(
            value
            for value in (chunk.text, chunk.symbol_name or "", chunk.file_path)
            if value
        )
        return tokenize_code(searchable)

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Build and persist an index from the exact chunks stored by indexing."""
        self._chunk_ids = [chunk.id for chunk in chunks]
        tokenized_documents = [self._document_tokens(chunk) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_documents) if tokenized_documents else None
        payload = {
            "repo": self._repo,
            "commit": self._commit,
            "chunker": self._chunker,
            "chunk_ids": self._chunk_ids,
            "documents": tokenized_documents,
        }
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def load(self) -> BM25Index:
        """Load a persisted index and reconstruct its rank_bm25 scorer."""
        if not self._path.is_file():
            raise FileNotFoundError(f"BM25 index does not exist: {self._path}")
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        expected = {"repo": self._repo, "commit": self._commit, "chunker": self._chunker}
        actual = {key: payload.get(key) for key in expected}
        if actual != expected:
            raise ValueError("BM25 index metadata does not match the requested repository")
        chunk_ids = payload.get("chunk_ids")
        documents = payload.get("documents")
        if not isinstance(chunk_ids, list) or not isinstance(documents, list):
            raise ValueError("BM25 index is missing chunk data")
        if len(chunk_ids) != len(documents):
            raise ValueError("BM25 index chunk IDs and documents have different lengths")
        if not all(isinstance(chunk_id, str) for chunk_id in chunk_ids):
            raise ValueError("BM25 index contains an invalid chunk ID")
        if not all(isinstance(document, list) for document in documents):
            raise ValueError("BM25 index contains invalid token data")
        self._chunk_ids = chunk_ids
        self._bm25 = BM25Okapi(documents) if documents else None
        return self

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` chunk IDs ranked by BM25 score."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if self._bm25 is None:
            return []
        query_tokens = tokenize_code(query)
        query_token_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (
                -len(query_token_set.intersection(self._bm25.doc_freqs[item[0]])),
                -float(item[1]),
                item[0],
            ),
        )
        return [
            (self._chunk_ids[index], float(score))
            for index, score in ranked[: min(k, len(self._chunk_ids))]
        ]

    @property
    def path(self) -> Path:
        """Return the on-disk index path."""
        return self._path
