from pathlib import Path

from src.graph.extract import extract_file, extract_files


def test_extracts_definitions_and_imports_with_line_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    file_path.write_text(
        "import os\n"
        "from collections import defaultdict as dd\n"
        "\n"
        "class Service:\n"
        "    def run(self):\n"
        "        return dd(list)\n"
        "\n"
        "def helper():\n"
        "    return os.getcwd()\n",
        encoding="utf-8",
    )

    result = extract_file(file_path, tmp_path)

    assert result["file_path"] == str(file_path)
    assert result["definitions"] == [
        {"name": "Service", "type": "class", "start_line": 4, "end_line": 6},
        {
            "name": "run",
            "type": "method",
            "start_line": 5,
            "end_line": 6,
            "parent_class": "Service",
        },
        {"name": "helper", "type": "function", "start_line": 8, "end_line": 9},
    ]
    assert result["imports"][0]["names"] == [{"name": "os", "alias": None}]
    assert result["imports"][1]["names"] == [{"name": "defaultdict", "alias": "dd"}]


def test_resolves_relative_imports_to_modules_and_siblings(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text("def helper(): pass\n", encoding="utf-8")
    source = package / "consumer.py"
    source.write_text(
        "from .helpers import helper\nfrom . import helpers\nfrom ..missing import value\n",
        encoding="utf-8",
    )

    imports = extract_file(source, tmp_path)["imports"]

    assert imports[0]["module"] == "helpers"
    assert imports[0]["level"] == 1
    assert imports[0]["resolved_path"] == str(package / "helpers.py")
    assert imports[1]["resolved_path"] == str(package / "helpers.py")
    assert imports[2]["resolved_path"] is None


def test_extract_files_returns_one_result_per_input(tmp_path: Path) -> None:
    paths = [tmp_path / "a.py", tmp_path / "b.py"]
    for path in paths:
        path.write_text("def run():\n    pass\n", encoding="utf-8")

    assert [item["file_path"] for item in extract_files(paths)] == [str(path) for path in paths]
