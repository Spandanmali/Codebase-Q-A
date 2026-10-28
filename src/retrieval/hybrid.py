"""Hybrid vector and BM25 retrieval with reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked chunk IDs using reciprocal rank fusion.

    Each unique ID contributes ``1 / (k + rank)`` for every ranking in which
    it appears, where ranks are one-based.  The default ``k=60`` is the
    conventional RRF constant: it reduces the effect of any single high rank
    while still rewarding agreement between retrievers.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    next_seen = 0
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = next_seen
                next_seen += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return [
        chunk_id
        for chunk_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], first_seen[item[0]]),
        )
    ]


class HybridRetriever:
    """Retrieve chunk IDs from vector and BM25 indexes and fuse their ranks."""

    def __init__(self, vector_store: Any, bm25_index: Any, embedder: Any) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._embedder = embedder

    def search(self, query: str, k: int = 5) -> list[str]:
        """Return the top ``k`` chunk IDs from both retrievers."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        vector = self._embedder.embed([query])[0]
        vector_results = self._vector_store.query(vector, 30)
        bm25_results = self._bm25_index.search(query, 30)
        vector_ids = [
            self._chunk_id(result)
            for result in vector_results
        ]
        bm25_ids = [self._chunk_id(result) for result in bm25_results]
        return rrf([vector_ids, bm25_ids])[:k]

    @staticmethod
    def _chunk_id(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            chunk_id = result.get("id")
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            chunk_id = result[0] if result else None
        else:
            chunk_id = None
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("retriever result must contain a non-empty string chunk ID")
        return chunk_id
