from pathlib import Path

from src.ingestion.overview import build_repo_overview


def test_build_repo_overview_returns_requested_document_content(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Example repository\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (tmp_path / "src" / "nested" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "src" / "nested" / "visible.py").write_text(
        "visible\n", encoding="utf-8"
    )
    (tmp_path / "src" / "nested" / "deep" / "hidden.py").write_text(
        "hidden\n", encoding="utf-8"
    )

    document = build_repo_overview(tmp_path)

    assert document["type"] == "overview"
    assert "# Example repository" in document["content"]
    assert "[project]" in document["content"]
    assert "pytest" in document["content"]
    assert "src/nested/visible.py" in document["content"]
    assert "src/nested/deep/hidden.py" not in document["content"]
