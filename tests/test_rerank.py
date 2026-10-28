from dataclasses import replace

from chunking.models import Chunk
from config import load_settings
from retrieval import rerank as rerank_module
from retrieval.rerank import Reranker


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        file_path=f"src/{chunk_id}.py",
        start_line=1,
        end_line=1,
        text=text,
    )


def test_rerank_scores_header_text_and_preserves_stable_ties(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, device: str) -> None:
            calls.append(("load", model_name, device))

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            calls.extend(("predict", query_text, text) for query_text, text in pairs)
            return [0.5, 0.9, 0.5]

    monkeypatch.setattr(rerank_module, "CrossEncoder", FakeCrossEncoder)
    reranker_module_load = rerank_module._load_model
    reranker_module_load.cache_clear()
    settings = replace(
        load_settings(),
        reranker_model="test-cross-encoder",
    )
    reranker = Reranker(settings=settings)
    chunks = [_chunk("first", "one"), _chunk("second", "two"), _chunk("third", "three")]

    result = reranker.rerank("find it", chunks, top_n=2)

    assert [chunk.id for chunk in result] == ["second", "first"]
    assert calls == [
        ("load", "test-cross-encoder", "cpu"),
        ("predict", "find it", "File: src/first.py | Parent class: None | "
         "Symbol name: None | Symbol type: None\none"),
        ("predict", "find it", "File: src/second.py | Parent class: None | "
         "Symbol name: None | Symbol type: None\ntwo"),
        ("predict", "find it", "File: src/third.py | Parent class: None | "
         "Symbol name: None | Symbol type: None\nthree"),
    ]


def test_empty_chunks_do_not_load_model(monkeypatch) -> None:
    monkeypatch.setattr(
        rerank_module,
        "CrossEncoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    rerank_module._load_model.cache_clear()

    assert Reranker(model_name="test-cross-encoder").rerank("query", [], 5) == []
