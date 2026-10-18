"""Secure, bounded cloning of GitHub repositories."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from config import settings

CLONE_BASE_DIR = settings.source_dir
CLONE_TIMEOUT_SECONDS = settings.clone_timeout_seconds
CLONE_MAX_SIZE_BYTES = settings.clone_max_size_bytes
_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9._-]+\Z")
_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")


class CloneError(RuntimeError):
    """Raised when a repository cannot be safely cloned."""


def _repository_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        raise ValueError("repository URL must be a non-empty string")

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"malformed repository URL: {error}") from error

    if (
        parsed.scheme != "https"
        or hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "repository URL must be https://github.com/<owner>/<repo>[.git] "
            "without credentials, ports, query strings, or fragments"
        )

    parts = parsed.path.split("/")
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError("repository URL must contain exactly an owner and repository")
    owner, repository = parts[1], parts[2]
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not repository or not _COMPONENT_PATTERN.fullmatch(owner):
        raise ValueError("repository owner contains invalid characters")
    if not _COMPONENT_PATTERN.fullmatch(repository):
        raise ValueError("repository name contains invalid characters")
    return f"https://github.com/{owner}/{repository}.git"


def _configured_integer(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _ensure_inside(path: Path, base: Path) -> None:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as error:
        raise CloneError("clone destination escaped the configured base directory") from error


def _clone_size(path: Path, base: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        _ensure_inside(root_path, base)
        for name in directories + files:
            entry = root_path / name
            _ensure_inside(entry, base)
            try:
                total += entry.lstat().st_size
            except OSError as error:
                raise CloneError(f"could not inspect cloned file {entry}") from error
    return total


def _run_git(
    arguments: list[str], destination: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            arguments,
            cwd=destination,
            env=environment,
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise CloneError(f"git operation timed out after {timeout} seconds") from error
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "no details returned").strip()
        if "redirect" in details.lower() or " 30" in details:
            details += " (the repository may have moved; redirects are disabled)"
        raise CloneError(f"git operation failed: {details}")
    return result


def clone_repo(
    url: str, dest_dir: str | os.PathLike[str], commit: str | None = None
) -> tuple[Path, str]:
    """Clone a GitHub repository into a fresh, bounded directory.

    ``dest_dir`` is retained for the public API but is deliberately not used as
    a filesystem path. The configured source directory is the only clone base.
    """
    repository_url = _repository_url(url)
    if commit is not None and (
        not isinstance(commit, str) or not _SHA_PATTERN.fullmatch(commit)
    ):
        raise ValueError("commit must be exactly a 40-character hexadecimal SHA")

    base_dir = Path(CLONE_BASE_DIR).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="repo-", dir=base_dir))
    _ensure_inside(destination, base_dir)
    timeout = _configured_integer("CLONE_TIMEOUT_SECONDS", CLONE_TIMEOUT_SECONDS)
    size_limit = _configured_integer("CLONE_MAX_SIZE_BYTES", CLONE_MAX_SIZE_BYTES)

    clone_command = [
        "git",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.https.allow=always",
        "-c",
        "http.followRedirects=false",
        "clone",
        "--depth",
        "1",
        "--filter=blob:limit=1m",
        "--no-recurse-submodules",
        "--",
        repository_url,
        str(destination),
    ]
    try:
        _run_git(clone_command, base_dir, timeout)
        if commit is not None:
            _run_git(
                [
                    "git",
                    "-c",
                    "protocol.allow=never",
                    "-c",
                    "protocol.https.allow=always",
                    "-c",
                    "http.followRedirects=false",
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    commit,
                ],
                destination,
                timeout,
            )
            _run_git(["git", "checkout", "--detach", commit], destination, timeout)

        size = _clone_size(destination, base_dir)
        if size > size_limit:
            raise CloneError(
                f"cloned repository is {size} bytes, exceeding the {size_limit}-byte limit"
            )
        result = _run_git(["git", "rev-parse", "HEAD"], destination, timeout)
        commit_hash = result.stdout.strip()
        if not _SHA_PATTERN.fullmatch(commit_hash):
            raise CloneError("git returned an invalid commit hash")
        return destination, commit_hash
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
