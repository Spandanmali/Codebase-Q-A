"""Safely discover text files that can be indexed from a repository."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import pathspec

DEFAULT_MAX_FILE_SIZE_BYTES: Final = 500 * 1024
DEFAULT_MAX_FILES: Final = 5_000

_LANGUAGES: Final = {
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".toml": "toml",
    ".yaml": "yaml",
}
_SKIPPED_DIRECTORIES: Final = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_LOCKFILE_NAMES: Final = {
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    "package-lock.json",
    "pipfile.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
    "yarn.lock",
}


class DiscoveryError(RuntimeError):
    """Raised when repository discovery cannot safely complete."""


def _has_control_character(path: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in path)


def _is_generated_or_lockfile(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in _LOCKFILE_NAMES
        or name.endswith(".lock")
        or ".min." in name
        or name.endswith("_pb2.py")
        or name.endswith("_generated.py")
        or name.endswith(".generated.py")
    )


def _is_test_file(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        "test" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )


def _load_gitignore(repo: Path) -> pathspec.PathSpec:
    gitignore = repo / ".gitignore"
    if not gitignore.is_file() or gitignore.is_symlink():
        return pathspec.PathSpec.from_lines("gitwildmatch", [])
    try:
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DiscoveryError(f"could not read {gitignore}: {error}") from error
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def discover_files(
    repo_path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> list[dict[str, str | int | bool]]:
    """Return safe, supported text files beneath ``repo_path``.

    Files excluded by standard repository directories, ``.gitignore``, file
    type, size, encoding, or safety checks are skipped. A ``DiscoveryError``
    is raised when the indexable file limit is exceeded.
    """
    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be greater than zero")
    if max_files <= 0:
        raise ValueError("max_files must be greater than zero")

    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise DiscoveryError(f"repository path is not a directory: {repo}")
    ignore_spec = _load_gitignore(repo)
    discovered: list[dict[str, str | int | bool]] = []

    for root, directories, filenames in os.walk(repo, topdown=True, followlinks=False):
        root_path = Path(root)
        directories[:] = [
            name
            for name in directories
            if name not in _SKIPPED_DIRECTORIES
            and not (root_path / name).is_symlink()
            and not _has_control_character(name)
        ]
        for filename in filenames:
            candidate = root_path / filename
            if candidate.is_symlink() or _has_control_character(filename):
                continue
            try:
                resolved = candidate.resolve()
                relative = resolved.relative_to(repo).as_posix()
            except (OSError, RuntimeError, ValueError) as error:
                raise DiscoveryError(f"could not resolve {candidate}: {error}") from error
            if ignore_spec.match_file(relative):
                continue
            language = _LANGUAGES.get(candidate.suffix.lower())
            if language is None or _is_generated_or_lockfile(candidate):
                continue
            try:
                size = candidate.stat().st_size
            except OSError as error:
                raise DiscoveryError(f"could not inspect {candidate}: {error}") from error
            if size > max_file_size_bytes:
                continue
            try:
                content = candidate.read_bytes()
                content.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if b"\x00" in content:
                continue
            discovered.append(
                {
                    "path": relative,
                    "language": language,
                    "size_bytes": size,
                    "is_test": _is_test_file(Path(relative)),
                    "is_doc": language == "markdown",
                }
            )
            if len(discovered) > max_files:
                raise DiscoveryError(
                    f"repository contains more than {max_files} indexable files"
                )
    return discovered
