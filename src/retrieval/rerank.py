"""Cross-encoder reranking for already retrieved chunks."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from functools import cache
from typing import Any

import psutil

try:
    from chunking.context import build_embedding_text
    from chunking.models import Chunk
    from config import Settings, load_settings
except ModuleNotFoundError:
    from config import Settings, load_settings

    from ..chunking.context import build_embedding_text
    from ..chunking.models import Chunk

try:
    from sentence_transformers import CrossEncoder
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency is absent
    CrossEncoder = None  # type: ignore[assignment,misc]


LOGGER = logging.getLogger(__name__)


def _memory_bytes() -> int:
    """Return this process's resident memory in bytes."""
    return psutil.Process().memory_info().rss


@cache
def _load_model(model_name: str) -> Any:
    """Load one CPU model per model name and record its load cost."""
    if CrossEncoder is None:
        raise ImportError(
            "sentence-transformers is required to use Reranker; "
            "install the project dependencies first"
        )

    started = time.perf_counter()
    memory_before = _memory_bytes()
    model = CrossEncoder(model_name, device="cpu")
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Loaded reranker model %s in %.3fs; memory %.2f MiB (delta %.2f MiB)",
        model_name,
        elapsed,
        _memory_bytes() / (1024 * 1024),
        (_memory_bytes() - memory_before) / (1024 * 1024),
    )
    return model


class Reranker:
    """Score retrieved chunks with a local sentence-transformers cross-encoder."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        active_settings = settings or load_settings()
        self._model_name = model_name or active_settings.reranker_model
        if not self._model_name.strip():
            raise ValueError("reranker model name must not be empty")

    @property
    def model_name(self) -> str:
        """Return the configured cross-encoder model name."""
        return self._model_name

    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_n: int,
    ) -> list[Chunk]:
        """Return up to ``top_n`` chunks ranked by cross-encoder score."""
        if top_n < 0:
            raise ValueError("top_n must not be negative")
        if not chunks or top_n == 0:
            return []

        pairs = [(query, build_embedding_text(chunk)) for chunk in chunks]
        started = time.perf_counter()
        scores = _load_model(self._model_name).predict(pairs)
        scores = [float(score) for score in scores]
        if len(scores) != len(chunks):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(chunks)} chunks"
            )

        ranked = sorted(
            enumerate(chunks),
            key=lambda item: (-scores[item[0]], item[0]),
        )
        elapsed = time.perf_counter() - started
        LOGGER.info(
            "Reranked %d chunks in %.3fs; memory %.2f MiB",
            len(chunks),
            elapsed,
            _memory_bytes() / (1024 * 1024),
        )
        return [chunk for _, chunk in ranked[:top_n]]
