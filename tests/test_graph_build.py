import json
from pathlib import Path

import networkx as nx
import pytest

from src.chunking.models import Chunk
from src.graph.build import build_graph, callees, callers, load_graph, neighbors, save_graph


def test_builds_typed_graph_and_links_chunks(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text(
        "class Service:\n"
        "    def finish(self):\n"
        "        pass\n"
        "    def run(self):\n"
        "        self.finish()\n"
        "\n"
        "def entry():\n"
        "    Service().run()\n",
        encoding="utf-8",
    )
    chunk = Chunk("chunk-1", str(path), 1, 8, path.read_text(encoding="utf-8"))

    graph = build_graph([path], tmp_path, [chunk])
    function = f"func:{path.resolve()}::Service.run"
    finish = f"func:{path.resolve()}::Service.finish"
    assert isinstance(graph, nx.DiGraph)
    assert graph.nodes[f"class:{path.resolve()}::Service"]["type"] == "class"
    assert graph.nodes[function]["chunk_ids"] == ["chunk-1"]
    assert graph[function][finish]["type"] == "calls"
    assert function in callers(graph, finish)
    assert finish in callees(graph, function)
    assert finish in neighbors(graph, function, 1)


def test_save_and_load_graph(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("def run():\n    pass\n", encoding="utf-8")
    graph = build_graph([path], tmp_path)
    saved = tmp_path / "graph.json"
    save_graph(graph, saved)
    loaded = load_graph(saved)
    assert set(loaded.nodes) == set(graph.nodes)
    assert set(loaded.edges) == set(graph.edges)
    assert json.loads(saved.read_text(encoding="utf-8"))["nodes"]


def test_adds_local_import_edges(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("def work():\n    pass\n", encoding="utf-8")
    consumer = tmp_path / "consumer.py"
    consumer.write_text("from helper import work\n", encoding="utf-8")

    graph = build_graph([helper, consumer], tmp_path)

    assert graph[f"file:{consumer.resolve()}"][f"file:{helper.resolve()}"]["type"] == "imports"


def test_rejects_invalid_line_ranges(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("x = 1\n", encoding="utf-8")
    graph = nx.DiGraph()
    graph.add_node("bad", type="file", file_path=str(path), start_line=1, end_line=2)
    with pytest.raises(ValueError, match="exceeds"):
        from src.graph.build import validate_graph

        validate_graph(graph, tmp_path)
