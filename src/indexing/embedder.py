"""Embedding backends and the local Ollama implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import Settings, load_settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot return valid vectors."""


class BaseEmbedder(ABC):
    """Interface implemented by embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the provider model name."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Return the vector size once it has been observed."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector for each input text."""


RequestFunction = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _ollama_request(host: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one non-streaming request to Ollama's embedding endpoint."""
    request = Request(
        f"{host.rstrip('/')}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_response = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise EmbeddingError(
            f"Ollama returned HTTP {error.code}: {detail or error.reason}"
        ) from error
    except (OSError, URLError, TimeoutError) as error:
        raise EmbeddingError(f"Could not connect to Ollama: {error}") from error

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise EmbeddingError("Ollama returned invalid JSON.") from error
    if not isinstance(result, dict):
        raise EmbeddingError("Ollama returned an unexpected response.")
    return result


class Embedder(BaseEmbedder):
    """Batch text embedding through a local Ollama model."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: Settings | None = None,
        host: str = "http://127.0.0.1:11434",
        batch_size: int = 32,
        retries: int = 2,
        timeout: float = 30.0,
        request: RequestFunction | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if retries < 0:
            raise ValueError("retries must not be negative")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        active_settings = settings or load_settings()
        self._model_name = model_name or active_settings.embedding_model
        if not self._model_name.strip():
            raise ValueError("embedding model name must not be empty")
        self._host = host
        self._batch_size = batch_size
        self._retries = retries
        self._timeout = timeout
        self._request = request or _ollama_request
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, preserving order and including empty strings."""
        if not texts:
            return []
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must contain only strings")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            result = self._embed_batch(batch)
            vectors.extend(result)
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self._model_name, "input": texts}
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._request(self._host, payload, self._timeout)
                vectors = self._parse_vectors(response, len(texts))
                self._validate_dimension(vectors)
                return vectors
            except (EmbeddingError, OSError, URLError, TimeoutError) as error:
                last_error = error
                if attempt == self._retries:
                    break

        raise EmbeddingError(
            f"Failed to embed a batch with model {self._model_name!r} "
            f"after {self._retries + 1} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _parse_vectors(response: dict[str, Any], expected_count: int) -> list[list[float]]:
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            raise EmbeddingError(
                f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 0} "
                f"vectors for {expected_count} texts."
            )

        vectors: list[list[float]] = []
        for vector in embeddings:
            if not isinstance(vector, list) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in vector
            ):
                raise EmbeddingError("Ollama returned an invalid embedding vector.")
            vectors.append([float(value) for value in vector])
        return vectors

    def _validate_dimension(self, vectors: list[list[float]]) -> None:
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingError(
                f"Ollama model {self._model_name!r} returned inconsistent vector "
                f"dimensions in one batch: {sorted(dimensions)}."
            )

        batch_dimension = dimensions.pop()
        if self._dimension is not None and batch_dimension != self._dimension:
            raise EmbeddingError(
                f"Ollama model {self._model_name!r} returned dimension "
                f"{batch_dimension}, expected {self._dimension}."
            )
        self._dimension = batch_dimension


EmbeddingBackend = BaseEmbedder
