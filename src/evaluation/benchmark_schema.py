"""Schema and loader for repository-grounded evaluation benchmarks."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["lookup", "multi-file", "conceptual", "unanswerable"]
CommitHash = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{40}$")]
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


class GoldRange(BaseModel):
    """An inclusive, one-based line range in a repository file."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def end_is_not_before_start(self) -> GoldRange:
        if self.end < self.start:
            raise ValueError("gold range end must be greater than or equal to start")
        return self


class BenchmarkItem(BaseModel):
    """A question and its source-of-truth locations at a pinned commit."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    commit: CommitHash
    question: str = Field(min_length=1)
    question_type: QuestionType
    gold_files: list[str]
    gold_ranges: list[GoldRange]
    reference_answer: str | None = None

    @model_validator(mode="after")
    def validate_gold_requirements(self) -> BenchmarkItem:
        if self.question_type == "unanswerable":
            if self.gold_files or self.gold_ranges:
                raise ValueError("unanswerable items must have empty gold_files and gold_ranges")
        elif not self.gold_files or not self.gold_ranges:
            raise ValueError("answerable items must have non-empty gold_files and gold_ranges")

        invalid_files = [path for path in self.gold_files if not _is_repository_path(path)]
        if invalid_files:
            raise ValueError(f"gold_files contain invalid repository paths: {invalid_files}")
        invalid_ranges = [
            item.file for item in self.gold_ranges if not _is_repository_path(item.file)
        ]
        if invalid_ranges:
            raise ValueError(f"gold_ranges contain invalid repository paths: {invalid_ranges}")
        range_files = {item.file for item in self.gold_ranges}
        if not range_files.issubset(self.gold_files):
            raise ValueError("every gold range file must be listed in gold_files")
        return self


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark file or its pinned repository evidence is invalid."""


def load_benchmarks(
    directory: str | Path | None = None,
    *,
    validate_repositories: bool = True,
) -> list[BenchmarkItem]:
    """Load and validate every YAML benchmark in ``directory``.

    Repository paths may be local paths or Git URLs. Remote repositories are cloned
    into a temporary directory only when repository evidence needs to be checked.
    """
    benchmark_dir = (
        Path(directory) if directory is not None else Path(__file__).parent / "benchmarks"
    )
    if not benchmark_dir.is_dir():
        raise BenchmarkValidationError(f"benchmark directory does not exist: {benchmark_dir}")

    items: list[BenchmarkItem] = []
    for path in sorted((*benchmark_dir.glob("*.yml"), *benchmark_dir.glob("*.yaml"))):
        items.extend(_load_file(path))

    if validate_repositories:
        repositories: dict[tuple[str, str], Path] = {}
        temporary_repositories: set[Path] = set()
        try:
            for item in items:
                key = (item.repo, item.commit.lower())
                if key not in repositories:
                    repositories[key] = _repository_at_commit(item.repo, item.commit)
                    if not _is_local_repository(item.repo):
                        temporary_repositories.add(repositories[key])
                _validate_repository_evidence(item, repositories[key])
        finally:
            for repository in temporary_repositories:
                shutil.rmtree(repository, ignore_errors=True)
    return items


def _load_file(path: Path) -> list[BenchmarkItem]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BenchmarkValidationError(f"could not read benchmark file {path}: {error}") from error

    records = raw if isinstance(raw, list) else [raw]
    if not records or any(not isinstance(record, dict) for record in records):
        raise BenchmarkValidationError(f"{path} must contain a benchmark mapping or list")

    try:
        return [BenchmarkItem.model_validate(record) for record in records]
    except ValueError as error:
        raise BenchmarkValidationError(f"invalid benchmark file {path}: {error}") from error


def _repository_at_commit(repo: str, commit: str) -> Path:
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise BenchmarkValidationError(f"commit must be exactly a 40-character SHA: {commit!r}")

    local_repo = Path(repo)
    if _is_local_repository(repo):
        _verify_commit(local_repo, commit)
        return local_repo

    temporary_directory = Path(tempfile.mkdtemp(prefix="benchmark-repo-"))
    try:
        _run_git(
            ["clone", "--no-checkout", "--filter=blob:none", "--", repo, str(temporary_directory)]
        )
        _run_git(["fetch", "--depth", "1", "origin", commit], temporary_directory)
        _verify_commit(temporary_directory, commit)
    except BenchmarkValidationError:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return temporary_directory


def _verify_commit(repository: Path, commit: str) -> None:
    result = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repository)
    if result.stdout.strip().lower() != commit.lower():
        raise BenchmarkValidationError(f"repository {repository} does not contain commit {commit}")


def _validate_repository_evidence(item: BenchmarkItem, repository: Path) -> None:
    for file in item.gold_files:
        _git_file_contents(repository, item.commit, file)
    for gold_range in item.gold_ranges:
        contents = _git_file_contents(repository, item.commit, gold_range.file)
        line_count = len(contents.splitlines())
        if gold_range.end > line_count:
            raise BenchmarkValidationError(
                f"{item.id}: {gold_range.file} has {line_count} lines at {item.commit}, "
                f"but range ends at {gold_range.end}"
            )


def _git_file_contents(repository: Path, commit: str, file: str) -> str:
    result = _run_git(["show", f"{commit}:{file}"], repository)
    return result.stdout


def _run_git(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BenchmarkValidationError(f"Git command failed: {' '.join(arguments)}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no details returned").strip()
        raise BenchmarkValidationError(f"Git command failed: {detail}")
    return result


def _is_repository_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _is_local_repository(repo: str) -> bool:
    path = Path(repo)
    return path.is_dir() and (path / ".git").exists()
