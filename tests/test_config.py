from pathlib import Path

import pytest

from config import load_settings


def test_defaults_are_local_and_project_relative(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.env")

    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.generation_model == "google/flan-t5-small"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.data_dir.name == "data"


def test_dotenv_values_are_loaded_and_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "CHUNK_SIZE=500\nCHUNK_OVERLAP=50\nGENERATION_MODEL='local-model'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHUNK_SIZE", "600")

    settings = load_settings(dotenv_path)

    assert settings.chunk_size == 600
    assert settings.chunk_overlap == 50
    assert settings.generation_model == "local-model"


def test_invalid_chunk_relationship_is_rejected(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("CHUNK_SIZE=10\nCHUNK_OVERLAP=10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        load_settings(dotenv_path)
