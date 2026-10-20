"""Build a compact overview document for a repository."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict


class OverviewDocument(TypedDict):
    """Document produced by :func:`build_repo_overview`."""

    type: str
    content: str


class OverviewError(RuntimeError):
    """Raised when a repository overview cannot be built."""


_OVERVIEW_FILES = ("README.md", "README.rst", "README", "pyproject.toml", "requirements.txt")


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise OverviewError(f"could not read {path}: {error}") from error


def _repository_tree(repo: Path, max_depth: int = 3) -> str:
    entries: list[str] = []
    for root, directories, filenames in os.walk(repo, topdown=True, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(repo)
        root_depth = len(relative_root.parts)
        directories[:] = sorted(
            directory
            for directory in directories
            if directory != ".git"
            and not (root_path / directory).is_symlink()
            and root_depth + 1 <= max_depth
        )
        for name in sorted(directories + filenames):
            entry = root_path / name
            relative = entry.relative_to(repo)
            if len(relative.parts) <= max_depth:
                entries.append(relative.as_posix())

    return "\n".join(entries) if entries else "(empty)"


def build_repo_overview(repo_path: str | os.PathLike[str]) -> OverviewDocument:
    """Build a document containing repository context for indexing."""
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise OverviewError(f"repository path is not a directory: {repo}")

    sections: list[str] = []
    readme = next(
        (repo / name for name in _OVERVIEW_FILES[:3] if (repo / name).is_file()),
        None,
    )
    if readme is not None:
        sections.append(f"README ({readme.name})\n{_read_file(readme)}")
    else:
        sections.append("README\n(not found)")

    dependency_files = _OVERVIEW_FILES[3:]
    for name in dependency_files:
        dependency_file = repo / name
        if dependency_file.is_file():
            sections.append(f"{name}\n{_read_file(dependency_file)}")
        else:
            sections.append(f"{name}\n(not found)")

    sections.append(f"Directory tree (depth <= 3)\n{_repository_tree(repo)}")
    return {"type": "overview", "content": "\n\n".join(sections)}
