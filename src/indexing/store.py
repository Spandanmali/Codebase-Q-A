"""Persistent Chroma vector storage for repository chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

try:
    from chunking.models import Chunk
except ModuleNotFoundError:
    from ..chunking.models import Chunk


class VectorStore:
    """Store and search chunks in a commit-specific Chroma collection."""

    def __init__(
        self,
        persist_dir: str | Path,
        repo: str,
        commit: str,
        chunker: str,
        embedding_model: str,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._repo = repo
        self._commit = commit
        self._chunker = chunker
        self._embedding_model = embedding_model
        self._collection_name = self._build_collection_name(
            repo, commit, chunker, embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "repo": repo,
                "commit": commit,
                "chunker": chunker,
                "embedding_model": embedding_model,
                "hnsw:space": "cosine",
            },
        )

    @staticmethod
    def _build_collection_name(
        repo: str, commit: str, chunker: str, embedding_model: str
    ) -> str:
        identity = "\0".join((repo, commit, chunker, embedding_model))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        readable = re.sub(r"[^A-Za-z0-9_-]+", "-", repo).strip("-_") or "repo"
        return f"vs-{readable[:42]}-{digest}"

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
        values = {
            key: value
            for key, value in vars(chunk).items()
            if value is not None
        }
        metadata: dict[str, str | int | float | bool] = {}
        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, sort_keys=True)
        return metadata

    @staticmethod
    def _dimension(vector: Sequence[float]) -> int:
        dimension = len(vector)
        if dimension == 0:
            raise ValueError("vectors must not be empty")
        return dimension

    def _stored_dimension(self) -> int | None:
        dimension = self._collection.metadata.get("dimension")
        return dimension if isinstance(dimension, int) and not isinstance(dimension, bool) else None

    def _validate_vectors(self, vectors: Sequence[Sequence[float]]) -> int | None:
        if not vectors:
            return None
        dimensions = {self._dimension(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("all vectors must have the same dimension")
        dimension = dimensions.pop()
        stored_dimension = self._stored_dimension()
        if stored_dimension is not None and dimension != stored_dimension:
            raise ValueError(
                f"vector dimension {dimension} does not match collection dimension "
                f"{stored_dimension}"
            )
        return dimension

    def add_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Upsert chunks and their embeddings by stable chunk ID."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        dimension = self._validate_vectors(vectors)
        if not chunks:
            return
        if dimension is not None and self._stored_dimension() is None:
            metadata = {
                key: value
                for key, value in self._collection.metadata.items()
                if key != "hnsw:space"
            }
            self._collection.modify(
                metadata={**metadata, "dimension": dimension}
            )
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=[list(vector) for vector in vectors],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def query(
        self,
        vector: Sequence[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks, optionally constrained by Chroma metadata filters."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        stored_dimension = self._stored_dimension()
        if stored_dimension is not None and len(vector) != stored_dimension:
            raise ValueError(
                f"vector dimension {len(vector)} does not match collection dimension "
                f"{stored_dimension}"
            )
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": result["ids"][0][index],
                "document": result["documents"][0][index],
                "metadata": result["metadatas"][0][index],
                "distance": result["distances"][0][index],
            }
            for index in range(len(result["ids"][0]))
        ]

    def query_repository(
        self,
        repo: str,
        vector: Sequence[float],
        k: int,
    ) -> list[dict[str, Any]]:
        """Search all indexed commits for a repository and return nearest chunks."""
        if not repo.strip():
            raise ValueError("repo must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        matches: list[dict[str, Any]] = []
        for collection_summary in self._client.list_collections():
            metadata = collection_summary.metadata or {}
            if (
                metadata.get("repo") != repo
                or metadata.get("embedding_model") != self._embedding_model
            ):
                continue
            collection = self._client.get_collection(collection_summary.name)
            if collection.count() == 0:
                continue
            result = collection.query(
                query_embeddings=[list(vector)],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            matches.extend(
                {
                    "id": result["ids"][0][index],
                    "document": result["documents"][0][index],
                    "metadata": result["metadatas"][0][index],
                    "distance": result["distances"][0][index],
                }
                for index in range(len(result["ids"][0]))
            )
        matches.sort(key=lambda item: item["distance"])
        return matches[:k]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._collection.count()

    def is_indexed(self) -> bool:
        """Return whether this repository/chunker combination completed indexing."""
        if self._collection.metadata.get("indexed") is True:
            return True
        # Collections created by older versions have no completion marker.
        return self.count() > 0

    def mark_indexed(self) -> None:
        """Mark the collection complete after all files have been processed."""
        metadata = {
            key: value
            for key, value in self._collection.metadata.items()
            if key != "hnsw:space"
        }
        metadata["indexed"] = True
        self._collection.modify(metadata=metadata)

    def delete_repo(self) -> None:
        """Delete every collection belonging to this repository."""
        for collection in self._client.list_collections():
            metadata = collection.metadata or {}
            if metadata.get("repo") == self._repo:
                self._client.delete_collection(collection.name)
