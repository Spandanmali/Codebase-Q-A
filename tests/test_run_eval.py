import json
from pathlib import Path

from src.evaluation.benchmark_schema import BenchmarkItem
from src.evaluation.run_eval import RetrievalConfig, run_evaluation

COMMIT = "a" * 40


def _item(item_id: str, question_type: str, question: str) -> BenchmarkItem:
    return BenchmarkItem.model_validate(
        {
            "id": item_id,
            "repo": "owner/repository",
            "commit": COMMIT,
            "question": question,
            "question_type": question_type,
            "gold_files": [] if question_type == "unanswerable" else ["src/a.py"],
            "gold_ranges": []
            if question_type == "unanswerable"
            else [{"file": "src/a.py", "start": 10, "end": 20}],
        }
    )


def test_run_evaluation_indexes_once_and_saves_aggregates(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    items = [
        _item("lookup-1", "lookup", "Find a"),
        _item("lookup-2", "lookup", "Find a again"),
        _item("unknown-1", "unanswerable", "Find missing"),
    ]
    index_calls = []

    class Settings:
        index_dir = tmp_path / "index"
        embedding_model = "embedding-model"
        generation_model = "generation-model"

    class FakeEmbedder:
        def __init__(self, *, settings) -> None:
            assert settings is not None

        def embed(self, questions):
            return [[float(len(questions[0]))]]

    class FakeStore:
        def __init__(self, *args) -> None:
            assert args[1:] == (
                "owner/repository",
                COMMIT,
                "fixed",
                "embedding-model",
            )

        def query(self, vector, k):
            return [
                {
                    "metadata": {
                        "file_path": "src/a.py",
                        "start_line": 15,
                        "end_line": 25,
                    }
                }
            ]

    monkeypatch.setattr("src.evaluation.run_eval.load_settings", lambda: Settings())
    monkeypatch.setattr("src.evaluation.run_eval.Embedder", FakeEmbedder)
    monkeypatch.setattr("src.evaluation.run_eval.VectorStore", FakeStore)
    monkeypatch.setattr(
        "src.evaluation.run_eval.index_repo",
        lambda source, **kwargs: index_calls.append((source, kwargs)),
    )

    result = run_evaluation(
        "fixed_vector",
        results_directory=tmp_path / "results",
        configs={"fixed_vector": RetrievalConfig(chunker="fixed", top_k=3)},
        benchmark_loader=lambda _directory: items,
    )

    assert len(index_calls) == 1
    assert index_calls[0][1] == {
        "chunker": "fixed",
        "commit": COMMIT,
        "allow_local": False,
    }
    assert result["commit_hashes"] == [COMMIT]
    assert result["models"] == {
        "embedding": "embedding-model",
        "generation": "generation-model",
    }
    assert result["metrics"]["lookup"]["count"] == 2
    assert result["metrics"]["lookup"]["file"]["recall_at_k"] == 1.0
    assert result["unanswerable"] == [
        {"id": "unknown-1", "question": "Find missing"}
    ]
    saved = json.loads((tmp_path / "results" / "fixed_vector.json").read_text())
    assert saved["config"] == "fixed_vector"
    assert "Question type" in capsys.readouterr().out
