from pathlib import Path

from src.graph.calls import extract_call_edges


def test_resolves_same_file_and_self_method_calls(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text(
        "def helper():\n"
        "    pass\n"
        "class Service:\n"
        "    def run(self):\n"
        "        helper()\n"
        "        self.finish()\n"
        "    def finish(self):\n"
        "        pass\n",
        encoding="utf-8",
    )

    edges = extract_call_edges([path], tmp_path)

    assert [edge["confidence"] for edge in edges] == ["resolved", "resolved"]
    assert edges[0]["target"].endswith("::helper")
    assert edges[1]["target"].endswith("::Service.finish")


def test_resolves_imported_name_and_marks_unknown_dynamic_calls(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("def work():\n    pass\n", encoding="utf-8")
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from helper import work\n"
        "def run(callback):\n"
        "    work()\n"
        "    callback()\n"
        "    getattr(callback, 'work')()\n",
        encoding="utf-8",
    )

    edges = extract_call_edges([helper, consumer], tmp_path)

    assert edges[0]["confidence"] == "resolved"
    assert edges[0]["target"].endswith("helper.py::work")
    assert [edge["confidence"] for edge in edges[1:]] == [
        "unresolved",
        "unresolved",
        "unresolved",
    ]


def test_lists_candidates_for_duplicate_names(tmp_path: Path) -> None:
    caller = tmp_path / "caller.py"
    caller.write_text(
        "def run():\n"
        "    work()\n",
        encoding="utf-8",
    )
    caller.write_text(
        "def work():\n"
        "    pass\n"
        "def work():\n"
        "    pass\n"
        "def run():\n"
        "    work()\n",
        encoding="utf-8",
    )

    edges = extract_call_edges([caller], tmp_path)

    assert edges[0]["confidence"] == "ambiguous"
    assert len(edges[0]["candidates"]) == 2
