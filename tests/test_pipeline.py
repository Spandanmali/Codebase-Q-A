from pathlib import Path

import pytest

from indexing import pipeline
from retrieval.bm25 import BM25Index


class FakeEmbedder:
    model_name = "test-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


@pytest.fixture
def pipeline_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = pipeline.load_settings()
    settings = settings.__class__(
        embedding_model="test-model",
        generation_model=settings.generation_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        data_dir=tmp_path,
        source_dir=tmp_path / "source",
        index_dir=tmp_path / "index",
        evaluation_dir=tmp_path / "evaluation",
        clone_timeout_seconds=settings.clone_timeout_seconds,
        clone_max_size_bytes=settings.clone_max_size_bytes,
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "Embedder", FakeEmbedder)
    return settings


def test_empty_repository_is_reported_and_marked_indexed(
    tmp_path: Path, pipeline_settings: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        stats = pipeline.index_repo(tmp_path, allow_local=True)

    assert stats["files"] == 0
    assert stats["skipped"] == 0
    assert stats["chunks"] == 0
    assert "empty or has no supported files" in caplog.text


def test_unreadable_file_is_skipped_without_stopping(
    tmp_path: Path, pipeline_settings: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "good.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("value = 2\n", encoding="utf-8")
    original_discover = pipeline.discover_files

    def discover_with_missing_file(path: Path) -> list[dict[str, str]]:
        result = original_discover(path)
        result.append({"path": "missing.py"})
        return result

    monkeypatch.setattr(pipeline, "discover_files", discover_with_missing_file)
    stats = pipeline.index_repo(tmp_path, allow_local=True)

    assert stats["files"] == 2
    assert stats["skipped"] == 1
    assert stats["chunks"] == 2


def test_indexing_same_repository_twice_does_not_duplicate(
    tmp_path: Path, pipeline_settings: object
) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")

    first = pipeline.index_repo(tmp_path, allow_local=True)
    second = pipeline.index_repo(tmp_path, allow_local=True)

    assert first["files"] == 1
    assert first["chunks"] == 1
    assert second["files"] == 0
    assert second["chunks"] == 1
    bm25 = BM25Index(
        pipeline_settings.index_dir,
        str(tmp_path.resolve()),
        "local",
        "ast",
    ).load()
    assert bm25.search("value", 1)[0][0].startswith("ast:")
