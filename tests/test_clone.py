from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ingestion import clone


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/octo-owner/repo", "https://github.com/octo-owner/repo.git"),
        ("https://github.com/octo.owner/repo.git", "https://github.com/octo.owner/repo.git"),
    ],
)
def test_valid_urls_are_normalized(url: str, expected: str) -> None:
    assert clone._repository_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/owner/repo",
        "https://gitlab.com/owner/repo",
        "https://user:pass@github.com/owner/repo",
        "https://github.com:443/owner/repo",
        "https://github.com/owner/repo?x=1",
        "https://github.com/owner/repo#fragment",
        "https://github.com/owner/repo/extra",
        "https://github.com/owner",
        "not a url",
        "https://github.com/owner/re; rm -rf",
        "https://github.com/owner/repo%2Fother",
    ],
)
def test_invalid_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        clone._repository_url(url)


def test_failed_clone_cleans_up_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clone, "CLONE_BASE_DIR", tmp_path)

    def fail(*args: object, **kwargs: object) -> SimpleNamespace:
        raise clone.CloneError("git operation failed: unavailable")

    monkeypatch.setattr(clone, "_run_git", fail)
    with pytest.raises(clone.CloneError):
        clone.clone_repo("https://github.com/owner/repo", "ignored")
    assert list(tmp_path.iterdir()) == []


def test_oversized_clone_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clone, "CLONE_BASE_DIR", tmp_path)
    monkeypatch.setenv("CLONE_MAX_SIZE_BYTES", "1")

    def fake_run(arguments: list[str], destination: Path, timeout: int) -> SimpleNamespace:
        if "clone" in arguments:
            clone_destination = Path(arguments[-1])
            (clone_destination / "file.txt").write_text("too large", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")

    monkeypatch.setattr(clone, "_run_git", fake_run)
    with pytest.raises(clone.CloneError, match="exceeding"):
        clone.clone_repo("https://github.com/owner/repo", "ignored")
    assert list(tmp_path.iterdir()) == []
