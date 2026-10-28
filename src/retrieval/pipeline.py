"""Composable retrieval stages used by evaluation and application callers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

try:
    from chunking.models import Chunk
    from retrieval.hybrid import rrf
except ModuleNotFoundError:
    from ..chunking.models import Chunk
    from .hybrid import rrf

LOGGER = logging.getLogger(__name__)

ChunkResult = dict[str, Any]
ChunkLoader = Callable[[str], ChunkResult | None]


class RetrievalPipeline:
    """Run configured retrieval stages through one comparable interface."""

    def __init__(
        self,
        *,
        vector_store: Any,
        embedder: Any | None = None,
        bm25_index: Any | None = None,
        reranker: Any | None = None,
        chunk_loader: ChunkLoader | None = None,
        vector: bool = False,
        bm25: bool = False,
        hybrid: bool = False,
        rerank: bool = False,
        graph_expansion: bool = False,
    ) -> None:
        if not (vector or bm25):
            raise ValueError("at least one of vector or bm25 must be enabled")
        if vector and embedder is None:
            raise ValueError("embedder is required when vector retrieval is enabled")
        if bm25 and bm25_index is None:
            raise ValueError("bm25_index is required when BM25 retrieval is enabled")
        if hybrid and not (vector and bm25):
            raise ValueError("hybrid fusion requires both vector and BM25 retrieval")
        if rerank and reranker is None:
            raise ValueError("reranker is required when reranking is enabled")

        self._vector_store = vector_store
        self._embedder = embedder
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._chunk_loader = chunk_loader
        self._vector = vector
        self._bm25 = bm25
        self._hybrid = hybrid
        self._rerank = rerank
        self._graph_expansion = graph_expansion

    def retrieve(self, query: str, top_k: int = 5) -> list[ChunkResult]:
        """Retrieve chunks using the configured stages."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidate_k = max(top_k, 30) if self._hybrid else top_k
        vector_results: list[ChunkResult] = []
        bm25_results: list[tuple[str, float]] = []

        if self._vector:
            started = time.perf_counter()
            vector = self._embedder.embed([query])[0]
            vector_results = self._vector_store.query(vector, candidate_k)
            self._log_stage("vector", started)

        if self._bm25:
            started = time.perf_counter()
            bm25_results = self._bm25_index.search(query, candidate_k)
            self._log_stage("bm25", started)

        if self._hybrid:
            started = time.perf_counter()
            ids = rrf(
                [
                    [self._result_id(result) for result in vector_results],
                    [chunk_id for chunk_id, _ in bm25_results],
                ]
            )[:candidate_k]
            results = self._load_results(ids, vector_results)
            self._log_stage("hybrid", started)
        else:
            results = self._combine_results(vector_results, bm25_results)

        if self._rerank:
            started = time.perf_counter()
            chunks = [self._as_chunk(result) for result in results]
            reranked = self._reranker.rerank(query, chunks, top_k)
            by_id = {result["id"]: result for result in results}
            results = [by_id[chunk.id] for chunk in reranked if chunk.id in by_id]
            self._log_stage("rerank", started)
        else:
            results = results[:top_k]

        if self._graph_expansion:
            started = time.perf_counter()
            self._log_stage("graph_expansion", started)

        return results

    def search(self, query: str, top_k: int = 5) -> list[ChunkResult]:
        """Alias for callers using the existing retriever naming convention."""
        return self.retrieve(query, top_k)

    def _combine_results(
        self,
        vector_results: Sequence[ChunkResult],
        bm25_results: Sequence[tuple[str, float]],
    ) -> list[ChunkResult]:
        if not bm25_results:
            return list(vector_results)
        results = list(vector_results)
        known_ids = {self._result_id(result) for result in results}
        for chunk_id, _ in bm25_results:
            if chunk_id in known_ids:
                continue
            result = self._load_result(chunk_id)
            if result is not None:
                results.append(result)
                known_ids.add(chunk_id)
        return results

    def _load_results(
        self, ids: Sequence[str], vector_results: Sequence[ChunkResult]
    ) -> list[ChunkResult]:
        by_id = {self._result_id(result): result for result in vector_results}
        results: list[ChunkResult] = []
        for chunk_id in ids:
            result = by_id.get(chunk_id) or self._load_result(chunk_id)
            if result is not None:
                results.append(result)
        return results

    def _load_result(self, chunk_id: str) -> ChunkResult | None:
        if self._chunk_loader is None:
            raise ValueError("chunk_loader is required for BM25 or hybrid results")
        result = self._chunk_loader(chunk_id)
        if result is not None and "id" not in result:
            result = {**result, "id": chunk_id}
        return result

    @staticmethod
    def _result_id(result: ChunkResult) -> str:
        chunk_id = result.get("id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("retrieval result must contain a non-empty string id")
        return chunk_id

    @staticmethod
    def _as_chunk(result: ChunkResult) -> Chunk:
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("retrieval result is missing metadata")
        return Chunk(
            id=RetrievalPipeline._result_id(result),
            file_path=str(metadata["file_path"]),
            start_line=int(metadata["start_line"]),
            end_line=int(metadata["end_line"]),
            text=str(result.get("document") or ""),
            language=metadata.get("language"),
            chunker=metadata.get("chunker"),
            symbol_name=metadata.get("symbol_name"),
            symbol_type=metadata.get("symbol_type"),
            parent_class=metadata.get("parent_class"),
            docstring=metadata.get("docstring"),
            context=metadata.get("context"),
        )

    @staticmethod
    def _log_stage(stage: str, started: float) -> None:
        LOGGER.info(
            "retrieval stage %s completed in %.3fs",
            stage,
            time.perf_counter() - started,
        )
