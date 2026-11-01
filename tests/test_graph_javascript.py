from pathlib import Path

from src.chunking.fixed import chunk_fixed
from src.graph.build import build_graph
from src.graph.extract import extract_file
from src.graph.parsers import get_parser

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sample_js_repo"


def test_extracts_javascript_typescript_definitions_and_imports() -> None:
    path = FIXTURE_ROOT / "components.tsx"
    result = extract_file(path, FIXTURE_ROOT)

    definitions = [
        (item["name"], item["type"], item["start_line"], item["end_line"])
        for item in result["definitions"]
    ]
    assert definitions == [
        ("App", "function", 6, 9),
        ("local", "function", 7, 7),
        ("Service", "class", 11, 15),
        ("run", "method", 12, 14),
        ("Widget", "function", 17, 17),
    ]
    assert result["imports"][0]["module"] == "react"
    assert result["imports"][0]["names"] == [
        {"name": "default", "alias": "React"},
        {"name": "useState", "alias": "state"},
    ]
    assert result["imports"][1]["resolved_path"] == str((FIXTURE_ROOT / "util.ts").resolve())
    assert result["imports"][3]["kind"] == "export"
    assert result["imports"][4]["resolved_path"] == str(
        (FIXTURE_ROOT / "nested" / "index.ts").resolve()
    )


def test_builds_javascript_edges_external_metadata_and_chunk_links() -> None:
    paths = [
        FIXTURE_ROOT / "components.tsx",
        FIXTURE_ROOT / "util.ts",
        FIXTURE_ROOT / "nested" / "index.ts",
    ]
    chunks = chunk_fixed(paths[0], paths[0].read_text(encoding="utf-8"), lines=5, overlap=1)
    graph = build_graph(paths, FIXTURE_ROOT, chunks)
    component = str(paths[0].resolve())

    assert f"func:{component}::App" in graph
    assert graph.nodes[f"func:{component}::App"]["chunk_ids"] == [
        f"fixed:{paths[0]}:5-9",
        f"fixed:{paths[0]}:9-13",
    ]
    assert graph.nodes[f"file:{component}"]["external_imports"] == ["express", "react"]
    assert not any(
        target.startswith("file:") and "node_modules" in target
        for target in graph.successors(f"file:{component}")
    )
    util_file = f"file:{(FIXTURE_ROOT / 'util.ts').resolve()}"
    assert graph[f"file:{component}"][util_file]["type"] == "imports"


def test_skips_syntax_errors_and_unsupported_extensions() -> None:
    assert extract_file(FIXTURE_ROOT / "broken.js", FIXTURE_ROOT)["definitions"] == []
    assert extract_file(FIXTURE_ROOT / "broken.js", FIXTURE_ROOT)["imports"] == []
    assert extract_file(FIXTURE_ROOT / "README.txt", FIXTURE_ROOT)["definitions"] == []
    assert get_parser("example.js") is not None
    assert get_parser("example.py") is None

    graph = build_graph(
        [FIXTURE_ROOT / "broken.js", FIXTURE_ROOT / "README.txt"],
        FIXTURE_ROOT,
    )
    assert graph.number_of_nodes() == 0
