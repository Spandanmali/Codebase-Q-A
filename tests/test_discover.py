from pathlib import Path

import pytest

from src.ingestion.discover import DiscoveryError, discover_files


def test_discover_files_filters_and_classifies_supported_text(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\nignored.json\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("assert True\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# README\n", encoding="utf-8")
    (tmp_path / "settings.toml").write_text("[tool]\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("enabled: true\n", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "hidden.py").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "generated_pb2.py").write_text("# generated\n", encoding="utf-8")

    result = discover_files(tmp_path)

    assert {entry["path"] for entry in result} == {
        "app.py",
        "tests/test_app.py",
        "README.md",
        "settings.toml",
        "config.yaml",
        "config.json",
    }
    readme = next(entry for entry in result if entry["path"] == "README.md")
    assert readme["language"] == "markdown"
    assert readme["is_doc"] is True
    assert next(entry for entry in result if entry["path"] == "tests/test_app.py")[
        "is_test"
    ] is True


def test_discover_files_supports_common_languages_and_skips_generated_content(
    tmp_path: Path,
) -> None:
    files = {
        "app.js": "const app = true;\n",
        "component.tsx": "export const Component = () => null;\n",
        "service.java": "class Service {}\n",
        "worker.go": "package worker\n",
        "main.rs": "fn main() {}\n",
        "query.sql": "select 1;\n",
        "page.html": "<html></html>\n",
        "styles.scss": "body { color: red; }\n",
        "Dockerfile": "FROM python:3.13\n",
        "Makefile": "all:\n",
        ".env.example": "DATABASE_URL=example\n",
        "notes.rst": "Notes\n=====\n",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value\n", encoding="utf-8")
    (tmp_path / "server.pem").write_text("private key\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dependency.js").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "types.d.ts").write_text("declare const value: string;\n", encoding="utf-8")
    (tmp_path / "app.min.js").write_text("minified\n", encoding="utf-8")

    result = discover_files(tmp_path)

    languages = {entry["path"]: entry["language"] for entry in result}
    assert languages == {
        "app.js": "javascript",
        "component.tsx": "typescript",
        "service.java": "java",
        "worker.go": "go",
        "main.rs": "rust",
        "query.sql": "sql",
        "page.html": "html",
        "styles.scss": "scss",
        "Dockerfile": "dockerfile",
        "Makefile": "makefile",
        ".env.example": "dotenv",
        "notes.rst": "rst",
    }


def test_discover_files_skips_symlink_binary_and_oversized_files(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-discovery.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    try:
        (tmp_path / "outside.py").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    (tmp_path / "binary.py").write_bytes(b"\x00\x01")
    (tmp_path / "invalid.py").write_bytes(b"\xff\xfe")
    (tmp_path / "large.py").write_text("x" * 20, encoding="utf-8")
    (tmp_path / "small.py").write_text("x\n", encoding="utf-8")

    result = discover_files(tmp_path, max_file_size_bytes=10)

    assert [entry["path"] for entry in result] == ["small.py"]


def test_discover_files_stops_after_file_limit(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file_{index}.py").write_text("x\n", encoding="utf-8")

    with pytest.raises(DiscoveryError, match="more than 2 indexable files"):
        discover_files(tmp_path, max_files=2)
