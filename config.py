"""Project settings loaded from a local .env file and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without requiring a dotenv dependency."""
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _setting(values: dict[str, str], name: str, default: str) -> str:
    """Use the process environment before the local .env file."""
    return os.environ.get(name, values.get(name, default))


def _integer(values: dict[str, str], name: str, default: int) -> int:
    raw_value = _setting(values, name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Configuration shared by the local ingestion and Q&A pipeline."""

    embedding_model: str
    generation_model: str
    chunk_size: int
    chunk_overlap: int
    data_dir: Path
    source_dir: Path
    index_dir: Path
    evaluation_dir: Path
    clone_timeout_seconds: int
    clone_max_size_bytes: int


def load_settings(dotenv_path: Path | None = None) -> Settings:
    """Load settings, allowing environment variables to override `.env` values."""
    env_path = dotenv_path or PROJECT_ROOT / ".env"
    values = _read_dotenv(env_path)

    data_dir = Path(_setting(values, "DATA_DIR", "data"))
    source_dir = Path(_setting(values, "SOURCE_DIR", str(data_dir / "source")))
    index_dir = Path(_setting(values, "INDEX_DIR", str(data_dir / "index")))
    evaluation_dir = Path(
        _setting(values, "EVALUATION_DIR", str(data_dir / "evaluation"))
    )

    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    if not index_dir.is_absolute():
        index_dir = PROJECT_ROOT / index_dir
    if not evaluation_dir.is_absolute():
        evaluation_dir = PROJECT_ROOT / evaluation_dir

    settings = Settings(
        embedding_model=_setting(
    values, "EMBEDDING_MODEL", "nomic-embed-text"
),
        generation_model=_setting(values, "GENERATION_MODEL", "google/flan-t5-small"),
        chunk_size=_integer(values, "CHUNK_SIZE", 800),
        chunk_overlap=_integer(values, "CHUNK_OVERLAP", 120),
        data_dir=data_dir,
        source_dir=source_dir,
        index_dir=index_dir,
        evaluation_dir=evaluation_dir,
        clone_timeout_seconds=_integer(values, "CLONE_TIMEOUT_SECONDS", 120),
        clone_max_size_bytes=_integer(
            values, "CLONE_MAX_SIZE_BYTES", 200 * 1024 * 1024
        ),
    )
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    return settings


settings = load_settings()


if __name__ == "__main__":
    for name, value in vars(settings).items():
        print(f"{name}={value}")
