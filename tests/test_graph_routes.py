from pathlib import Path

from src.graph.routes import extract_route_graph, extract_routes


def test_extracts_fastapi_and_flask_routes(tmp_path: Path) -> None:
    path = tmp_path / "routes.py"
    path.write_text(
        '@app.get("/users")\n'
        "def users():\n"
        "    pass\n"
        '\n@router.api_route("/items", methods=["GET", "POST"])\n'
        "def items():\n"
        "    pass\n"
        '\n@blueprint.route("/health")\n'
        "def health():\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert extract_routes([path]) == [
        {"method": "GET", "path": "/users", "handler": "users", "file": str(path), "line": 1},
        {"method": "GET", "path": "/items", "handler": "items", "file": str(path), "line": 5},
        {"method": "POST", "path": "/items", "handler": "items", "file": str(path), "line": 5},
        {"method": "GET", "path": "/health", "handler": "health", "file": str(path), "line": 9},
    ]


def test_route_graph_links_route_nodes_to_handlers(tmp_path: Path) -> None:
    path = tmp_path / "routes.py"
    path.write_text('@app.post("/users")\ndef create_user():\n    pass\n', encoding="utf-8")

    graph = extract_route_graph([path])

    route_node = next(node for node in graph["nodes"] if node["type"] == "route")
    handler_node = next(node for node in graph["nodes"] if node["type"] == "function")
    assert route_node["method"] == "POST"
    assert route_node["path"] == "/users"
    assert route_node["handler"] == "create_user"
    assert graph["edges"] == [
        {"source": route_node["id"], "target": handler_node["id"], "type": "route"}
    ]
