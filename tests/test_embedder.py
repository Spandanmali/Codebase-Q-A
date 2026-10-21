from dataclasses import replace

import pytest

from config import load_settings
from indexing.embedder import Embedder, EmbeddingBackend, EmbeddingError


def test_embed_batches_requests_and_preserves_empty_strings() -> None:
    calls: list[dict[str, object]] = []

    def request(host: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        calls.append(payload)
        texts = payload["input"]
        assert isinstance(texts, list)
        return {"embeddings": [[float(len(text)), 1.0] for text in texts]}

    embedder = Embedder(
        model_name="test-model",
        batch_size=2,
        request=request,
    )

    assert embedder.embed(["", "hello", "world"]) == [
        [0.0, 1.0],
        [5.0, 1.0],
        [5.0, 1.0],
    ]
    assert [call["input"] for call in calls] == [["", "hello"], ["world"]]
    assert embedder.model_name == "test-model"
    assert embedder.dimension == 2


def test_empty_input_does_not_call_ollama() -> None:
    def request(*_args: object) -> dict[str, object]:
        pytest.fail("request should not be called")
        return {}

    embedder = Embedder(model_name="test-model", request=request)

    assert embedder.embed([]) == []
    assert embedder.dimension is None


def test_retries_then_reports_model_name() -> None:
    attempts = 0

    def request(*_args: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        raise EmbeddingError("temporary failure")

    embedder = Embedder(model_name="retry-model", retries=2, request=request)

    with pytest.raises(EmbeddingError, match="retry-model"):
        embedder.embed(["text"])
    assert attempts == 3


def test_inconsistent_dimensions_are_rejected() -> None:
    def request(*_args: object) -> dict[str, object]:
        return {"embeddings": [[1.0, 2.0], [3.0]]}

    embedder = Embedder(model_name="dimension-model", request=request)

    with pytest.raises(EmbeddingError, match="inconsistent"):
        embedder.embed(["one", "two"])


def test_model_defaults_to_config() -> None:
    settings = load_settings()
    custom_settings = replace(settings, embedding_model="configured-model")
    embedder = Embedder(settings=custom_settings, request=lambda *_args: {"embeddings": []})

    assert embedder.model_name == "configured-model"


def test_backend_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingBackend()
