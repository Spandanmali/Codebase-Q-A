"""Extract FastAPI and Flask route decorators and their handler links."""

import ast
from pathlib import Path
from typing import Any

import tree_sitter_python
from tree_sitter import Language, Node, Parser

_PARSER = Parser(Language(tree_sitter_python.language()))
_METHOD_DECORATORS = {
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "trace",
}
_ROUTE_DECORATORS = {"api_route", "route"}


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _literal(node: Node | None, source: bytes) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(_text(node, source))
    except (SyntaxError, ValueError):
        return None


def _call(node: Node) -> Node | None:
    if node.type == "call":
        return node
    return next((child for child in node.named_children if child.type == "call"), None)


def _decorator_name(call: Node, source: bytes) -> str | None:
    function = call.child_by_field_name("function")
    if function is None or function.type != "attribute":
        return None
    attribute = function.child_by_field_name("attribute")
    return _text(attribute, source) if attribute is not None else None


def _arguments(call: Node) -> list[Node]:
    arguments = call.child_by_field_name("arguments")
    return list(arguments.named_children) if arguments is not None else []


def _route_values(
    call: Node, decorator_name: str, source: bytes
) -> tuple[str, str, list[str]] | None:
    arguments = _arguments(call)
    path_node = next(
        (argument for argument in arguments if argument.type != "keyword_argument"), None
    )
    path = _literal(path_node, source)
    if not isinstance(path, str):
        return None

    methods: list[str] = []
    methods_node = next(
        (
            argument
            for argument in arguments
            if argument.type == "keyword_argument"
            and _text(argument.child_by_field_name("name"), source) == "methods"
        ),
        None,
    )
    if methods_node is not None:
        value = _literal(methods_node.child_by_field_name("value"), source)
        if isinstance(value, str):
            methods = [value.upper()]
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            methods = [item.upper() for item in value]

    if decorator_name in _METHOD_DECORATORS:
        methods = [decorator_name.upper()]
    elif not methods:
        methods = ["GET"]
    return decorator_name, path, methods


def _handler_name(
    definition: Node, source: bytes, class_stack: tuple[str, ...]
) -> tuple[str, str]:
    name_node = definition.child_by_field_name("name")
    name = _text(name_node, source) if name_node is not None else ""
    symbol = ".".join((*class_stack, name))
    return name, symbol


def _extract_from_node(
    node: Node,
    source: bytes,
    file_path: str,
    class_stack: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if node.type == "decorated_definition":
        definition = next(
            (
                child
                for child in node.named_children
                if child.type in {"function_definition", "class_definition"}
            ),
            None,
        )
        if definition is None:
            return []
        name, symbol = _handler_name(definition, source, class_stack)
        if definition.type != "function_definition":
            nested = definition.child_by_field_name("body")
            return (
                _extract_from_node(nested, source, file_path, (*class_stack, name))
                if nested is not None
                else []
            )
        routes: list[dict[str, Any]] = []
        for decorator in (child for child in node.named_children if child.type == "decorator"):
            call = _call(decorator)
            if call is None:
                continue
            decorator_name = _decorator_name(call, source)
            if decorator_name not in _METHOD_DECORATORS | _ROUTE_DECORATORS:
                continue
            values = _route_values(call, decorator_name, source)
            if values is None:
                continue
            _, path, methods = values
            for method in methods:
                routes.append(
                    {
                        "method": method,
                        "path": path,
                        "handler": symbol,
                        "file": file_path,
                        "line": decorator.start_point.row + 1,
                    }
                )
        return routes

    if node.type == "class_definition":
        name_node = node.child_by_field_name("name")
        name = _text(name_node, source) if name_node is not None else ""
        body = node.child_by_field_name("body")
        return (
            _extract_from_node(body, source, file_path, (*class_stack, name))
            if body is not None
            else []
        )

    routes = []
    for child in node.named_children:
        routes.extend(_extract_from_node(child, source, file_path, class_stack))
    return routes


def extract_routes(
    file_paths: list[str | Path], repo_root: str | Path | None = None
) -> list[dict[str, Any]]:
    """Return route metadata found in the supplied Python files."""
    del repo_root
    routes: list[dict[str, Any]] = []
    for file_path in file_paths:
        path = Path(file_path)
        source = path.read_bytes()
        root = _PARSER.parse(source).root_node
        routes.extend(_extract_from_node(root, source, str(path)))
    return routes


def extract_route_graph(
    file_paths: list[str | Path], repo_root: str | Path | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Return route nodes, handler nodes, and route-to-handler edges."""
    del repo_root
    routes = extract_routes(file_paths)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    handler_nodes: dict[str, dict[str, Any]] = {}
    for route in routes:
        handler_id = f"{route['file']}::{route['handler']}"
        route_id = (
            f"{route['file']}::route:{route['line']}:{route['method']}:{route['path']}"
        )
        nodes.append({"id": route_id, "type": "route", **route})
        if handler_id not in handler_nodes:
            handler_nodes[handler_id] = {
                "id": handler_id,
                "type": "function",
                "symbol": route["handler"],
                "file": route["file"],
            }
        edges.append({"source": route_id, "target": handler_id, "type": "route"})
    nodes.extend(handler_nodes.values())
    return {"nodes": nodes, "edges": edges}


__all__ = ["extract_route_graph", "extract_routes"]
