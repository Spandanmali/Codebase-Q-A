"""Build and query a NetworkX graph of the repository's Python code."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx

from .calls import extract_call_edges
from .extract import extract_files
from .routes import extract_routes

_NODE_TYPES = {"file", "function", "class", "route"}
_EDGE_TYPES = {"imports", "defines", "calls", "handled_by"}


def _path(value: str | Path, repo_root: Path) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return str(candidate.resolve())


def _chunk_values(chunks: Iterable[Any] | None) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for chunk in chunks or ():
        if isinstance(chunk, Mapping):
            result.append(chunk)
        else:
            result.append(vars(chunk))
    return result


def _attach_chunks(
    graph: nx.DiGraph, chunks: Iterable[Any] | None, repo_root: Path
) -> None:
    chunk_values = _chunk_values(chunks)
    for node, attributes in graph.nodes(data=True):
        file_path = attributes["file_path"]
        start = attributes["start_line"]
        end = attributes["end_line"]
        matching = []
        for chunk in chunk_values:
            chunk_path = _path(str(chunk["file_path"]), repo_root)
            if (
                chunk_path == file_path
                and int(chunk["start_line"]) <= end
                and int(chunk["end_line"]) >= start
            ):
                matching.append(str(chunk["id"]))
        attributes["chunk_ids"] = sorted(set(matching))


def _definition_id(file_path: str, definition: Mapping[str, Any]) -> str:
    kind = "class" if definition["type"] == "class" else "func"
    symbol = definition["name"]
    if definition.get("parent_class"):
        symbol = f"{definition['parent_class']}.{symbol}"
    return f"{kind}:{file_path}::{symbol}"


def _call_id(identifier: str, kind: str = "func") -> str:
    file_path, separator, symbol = identifier.partition("::")
    if not separator:
        return identifier
    return f"{kind}:{file_path}::{symbol}"


def _absolute_import_path(import_data: Mapping[str, Any], repo_root: Path) -> str | None:
    module = str(import_data.get("module") or "")
    if not module and import_data.get("kind") == "import":
        names = import_data.get("names") or []
        module = str(names[0].get("name", "")).split(".", 1)[0] if names else ""
    if not module:
        return None
    base = repo_root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def build_graph(
    file_paths: Iterable[str | Path],
    repo_root: str | Path,
    chunks: Iterable[Any] | None = None,
) -> nx.DiGraph:
    """Build a graph from Python files and optionally link nodes to chunks."""
    root = Path(repo_root).resolve()
    paths = [Path(path).resolve() for path in file_paths]
    graph = nx.DiGraph()
    extracted = extract_files(paths, root)
    definitions: dict[str, dict[str, Any]] = {}

    for result in extracted:
        file_path = _path(result["file_path"], root)
        line_count = len(Path(file_path).read_text(encoding="utf-8").splitlines()) or 1
        file_id = f"file:{file_path}"
        graph.add_node(
            file_id,
            type="file",
            file_path=file_path,
            start_line=1,
            end_line=line_count,
            chunk_ids=[],
        )
        for definition in result["definitions"]:
            definition = {**definition, "file_path": file_path}
            node_id = _definition_id(file_path, definition)
            symbol = definition["name"]
            if definition.get("parent_class"):
                symbol = f"{definition['parent_class']}.{symbol}"
            definitions[f"{file_path}::{symbol}"] = definition
            graph.add_node(
                node_id,
                type="class" if definition["type"] == "class" else "function",
                file_path=file_path,
                start_line=definition["start_line"],
                end_line=definition["end_line"],
                chunk_ids=[],
                symbol=(
                    f"{definition['parent_class']}.{definition['name']}"
                    if definition.get("parent_class")
                    else definition["name"]
                ),
            )
            graph.add_edge(file_id, node_id, type="defines")
            if definition.get("parent_class"):
                class_id = f"class:{file_path}::{definition['parent_class']}"
                if class_id in graph:
                    graph.add_edge(class_id, node_id, type="defines")

        for import_data in result["imports"]:
            resolved = import_data.get("resolved_path") or _absolute_import_path(import_data, root)
            if resolved:
                target = f"file:{_path(resolved, root)}"
                if target in graph:
                    graph.add_edge(file_id, target, type="imports")

    for call in extract_call_edges(paths, root):
        if call.get("target") is None:
            continue
        source = _call_id(str(call["source"]))
        target = _call_id(str(call["target"]))
        if source in graph and target in graph:
            graph.add_edge(source, target, type="calls")

    for route in extract_routes(paths, root):
        file_path = _path(route["file"], root)
        handler_symbol = route["handler"]
        handler_key = f"{file_path}::{handler_symbol}"
        handler = definitions.get(handler_key)
        if handler is None:
            continue
        handler_id = _definition_id(file_path, handler)
        route_id = f"route:{file_path}::{route['method']}:{route['path']}:{route['line']}"
        graph.add_node(
            route_id,
            type="route",
            file_path=file_path,
            start_line=route["line"],
            end_line=handler["end_line"],
            method=route["method"],
            path=route["path"],
            handler=handler_symbol,
            chunk_ids=[],
        )
        graph.add_edge(route_id, handler_id, type="handled_by")

    _attach_chunks(graph, chunks, root)
    validate_graph(graph, root)
    return graph


def validate_graph(graph: nx.DiGraph, repo_root: str | Path) -> None:
    """Raise ``ValueError`` when node types, locations, or edges are invalid."""
    root = Path(repo_root).resolve()
    for node, attributes in graph.nodes(data=True):
        if attributes.get("type") not in _NODE_TYPES:
            raise ValueError(f"invalid node type for {node!r}")
        file_path = Path(str(attributes.get("file_path", "")))
        try:
            file_path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"node {node!r} is outside the repository") from error
        if not file_path.is_file():
            raise ValueError(f"node {node!r} references missing file {file_path}")
        start = attributes.get("start_line")
        end = attributes.get("end_line")
        line_count = len(file_path.read_text(encoding="utf-8").splitlines())
        if not isinstance(start, int) or not isinstance(end, int) or not (1 <= start <= end):
            raise ValueError(f"node {node!r} has an invalid line range")
        if end > line_count:
            raise ValueError(f"node {node!r} line range exceeds {file_path}")
    for source, target, attributes in graph.edges(data=True):
        if attributes.get("type") not in _EDGE_TYPES:
            raise ValueError(f"invalid edge type for {source!r}->{target!r}")


def save_graph(graph: nx.DiGraph, path: str | Path) -> None:
    """Save a graph as portable JSON."""
    payload = {
        "nodes": [{"id": node, **attributes} for node, attributes in graph.nodes(data=True)],
        "edges": [
            {"source": source, "target": target, **attributes}
            for source, target, attributes in graph.edges(data=True)
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_graph(path: str | Path) -> nx.DiGraph:
    """Load a graph saved by :func:`save_graph`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    graph.add_nodes_from((item["id"], {key: value for key, value in item.items() if key != "id"})
                         for item in payload["nodes"])
    graph.add_edges_from(
        (
            item["source"],
            item["target"],
            {key: value for key, value in item.items() if key not in {"source", "target"}},
        )
        for item in payload["edges"]
    )
    return graph


def callers(graph: nx.DiGraph, node: str) -> set[str]:
    """Return nodes that call ``node``."""
    return {
        source
        for source, _, data in graph.in_edges(node, data=True)
        if data.get("type") == "calls"
    }


def callees(graph: nx.DiGraph, node: str) -> set[str]:
    """Return nodes called by ``node``."""
    return {
        target
        for _, target, data in graph.out_edges(node, data=True)
        if data.get("type") == "calls"
    }


def neighbors(graph: nx.DiGraph, node: str, depth: int = 1) -> set[str]:
    """Return nodes reachable in either direction within ``depth`` edges."""
    if depth < 0:
        raise ValueError("depth must not be negative")
    if node not in graph or depth == 0:
        return set()
    undirected = graph.to_undirected()
    return set(nx.single_source_shortest_path_length(undirected, node, cutoff=depth)) - {node}


__all__ = [
    "build_graph",
    "callees",
    "callers",
    "load_graph",
    "neighbors",
    "save_graph",
    "validate_graph",
]
