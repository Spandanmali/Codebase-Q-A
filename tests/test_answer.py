import pytest

from generation import answer as answer_module


class FakeEmbedder:
    def __init__(self, settings: object) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, *args: object) -> None:
        pass

    def query_repository(
        self, repo: str, vector: list[float], k: int
    ) -> list[dict[str, object]]:
        return []


def test_empty_retrieval_does_not_call_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(answer_module, "VectorStore", FakeStore)
    monkeypatch.setattr(
        answer_module,
        "_ollama_chat_request",
        lambda *_args: pytest.fail("LLM should not be called"),
    )

    assert answer_module.answer("What is this?", "repo") == ("not enough information", [])


def test_llm_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class StoreWithChunk(FakeStore):
        def query_repository(
            self, repo: str, vector: list[float], k: int
        ) -> list[dict[str, object]]:
            return [
                {
                    "document": "value = 1",
                    "metadata": {"file_path": "config.py", "start_line": 1, "end_line": 1},
                }
            ]

    monkeypatch.setattr(answer_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(answer_module, "VectorStore", StoreWithChunk)
    monkeypatch.setattr(
        answer_module,
        "_ollama_chat_request",
        lambda *_args: (_ for _ in ()).throw(answer_module.AnswerError("timed out")),
    )

    with pytest.raises(answer_module.AnswerError, match="timed out"):
        answer_module.answer("What is the value?", "repo")
