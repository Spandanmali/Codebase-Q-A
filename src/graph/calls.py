"""Extract and conservatively resolve Python call edges with tree-sitter.

Resolution is intentionally lexical: it cannot prove runtime rebinding,
inheritance dispatch, decorator effects, callback targets, or values returned
by ``getattr``. Those cases are reported as unresolved instead of guessed.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from src.graph.extract import _candidate_path, _module_name

_PARSER = Parser(Language(tree_sitter_python.language()))


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _name(node: Node | None, source: bytes) -> str:
    return _text(node, source) if node is not None else ""


@dataclass(frozen=True)
class _Definition:
    file_path: str
    name: str
    kind: str
    parent_class: str | None
    node: Node

    @property
    def qualified_name(self) -> str:
        return f"{self.parent_class}.{self.name}" if self.parent_class else self.name

    @property
    def identifier(self) -> str:
        return f"{self.file_path}::{self.qualified_name}"


def _definition_node(node: Node) -> Node | None:
    if node.type in {"function_definition", "class_definition"}:
        return node
    if node.type == "decorated_definition":
        return next(
            (
                child
                for child in node.named_children
                if child.type in {"function_definition", "class_definition"}
            ),
            None,
        )
    return None


def _definitions(
    node: Node,
    source: bytes,
    file_path: str,
    class_stack: tuple[str, ...] = (),
) -> list[_Definition]:
    result: list[_Definition] = []
    definition = _definition_node(node)
    if definition is not None:
        name = _name(definition.child_by_field_name("name"), source)
        kind = "class" if definition.type == "class_definition" else (
            "method" if class_stack else "function"
        )
        parent_class = class_stack[-1] if kind == "method" else None
        result.append(_Definition(file_path, name, kind, parent_class, definition))
        body = definition.child_by_field_name("body")
        if body is not None:
            next_classes = class_stack + ((name,) if kind == "class" else ())
            for child in body.named_children:
                result.extend(_definitions(child, source, file_path, next_classes))
        return result

    for child in node.named_children:
        result.extend(_definitions(child, source, file_path, class_stack))
    return result


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _module_path(file_path: Path, repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        return ".".join(_module_name(file_path, repo_root))
    except ValueError:
        return None


def _import_bindings(
    root: Node,
    source: bytes,
    file_path: Path,
    repo_root: Path | None,
    known_files: set[str],
) -> dict[str, tuple[str, str | None]]:
    """Return local binding -> (module file, imported symbol or None)."""
    bindings: dict[str, tuple[str, str | None]] = {}
    for node in _walk(root):
        if node.type == "import_statement":
            for child in node.named_children:
                value = _text(child, source)
                imported, _, alias = value.partition(" as ")
                local = alias or imported.split(".")[0]
                module = imported
                module_path = (
                    _candidate_path(tuple(module.split(".")), repo_root) if repo_root else None
                )
                if module_path is None:
                    continue
                bindings[local] = (module_path, None)
        elif node.type == "import_from_statement":
            relative = next(
                (child for child in node.named_children if child.type == "relative_import"),
                None,
            )
            if relative is not None:
                relative_text = _text(relative, source)
                level = len(relative_text) - len(relative_text.lstrip("."))
                module = relative_text[level:]
            else:
                level = 0
                module_node = next(
                    (
                        child
                        for child in node.named_children
                        if child.type not in {"import", "from", "relative_import"}
                    ),
                    None,
                )
                module = _text(module_node, source) if module_node else ""
            module_path: str | None = None
            if repo_root is not None:
                if level:
                    try:
                        current = list(_module_name(file_path, repo_root))
                        package = current if file_path.name == "__init__.py" else current[:-1]
                        base = package[: len(package) - level + 1]
                        module_path = _candidate_path(
                            tuple(base) + tuple(module.split(".")), repo_root
                        )
                    except ValueError:
                        module_path = None
                else:
                    module_path = _candidate_path(tuple(module.split(".")), repo_root)
            if module_path is None:
                continue
            for child in node.named_children:
                if child.type not in {"dotted_name", "aliased_import"}:
                    continue
                value = _text(child, source)
                imported, _, alias = value.partition(" as ")
                local = alias or imported
                bindings[local] = (module_path, imported)
    return {name: binding for name, binding in bindings.items() if binding[0] in known_files}


def _walk_calls(node: Node) -> Iterable[Node]:
    for child in node.named_children:
        if child.type in {
            "function_definition",
            "class_definition",
            "decorated_definition",
            "lambda",
        }:
            continue
        yield child
        yield from _walk_calls(child)


def _call_name(node: Node, source: bytes) -> tuple[str, str | None]:
    function = node.child_by_field_name("function")
    if function is None:
        return "", None
    if function.type == "identifier":
        return _text(function, source), "name"
    if function.type == "attribute":
        object_node = function.child_by_field_name("object")
        attribute = function.child_by_field_name("attribute")
        return (
            f"{_text(object_node, source)}.{_text(attribute, source)}",
            (
                "self"
                if object_node is not None and _text(object_node, source) == "self"
                else "attribute"
            ),
        )
    return _text(function, source), "dynamic"


def _edge(
    caller: _Definition,
    call_node: Node,
    call_name: str,
    confidence: str,
    candidates: list[_Definition],
) -> dict[str, Any]:
    candidate_ids = [candidate.identifier for candidate in candidates]
    return {
        "source": caller.identifier,
        "target": candidate_ids[0] if confidence == "resolved" else None,
        "call": call_name,
        "confidence": confidence,
        "candidates": candidate_ids,
        "file_path": caller.file_path,
        "start_line": call_node.start_point.row + 1,
        "end_line": call_node.end_point.row + 1,
    }


def _extract_edges(
    file_path: Path,
    source: bytes,
    root: Node,
    all_definitions: list[_Definition],
    known_files: set[str],
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    local_definitions = [item for item in all_definitions if item.file_path == str(file_path)]
    bindings = _import_bindings(root, source, file_path, repo_root, known_files)
    edges: list[dict[str, Any]] = []
    for caller in local_definitions:
        if caller.kind == "class":
            continue
        body = caller.node.child_by_field_name("body")
        if body is None:
            continue
        for call_node in _walk_calls(body):
            if call_node.type != "call":
                continue
            call_name, call_kind = _call_name(call_node, source)
            candidates: list[_Definition] = []
            if call_kind == "name":
                if call_name == "getattr":
                    confidence = "unresolved"
                else:
                    candidates = [
                        item
                        for item in all_definitions
                        if item.file_path == str(file_path) and item.name == call_name
                    ]
                    if not candidates and call_name in bindings:
                        imported_file, imported_name = bindings[call_name]
                        if imported_name:
                            candidates = [
                                item
                                for item in all_definitions
                                if item.file_path == imported_file and item.name == imported_name
                            ]
                    confidence = (
                        "resolved" if len(candidates) == 1
                        else "ambiguous" if candidates
                        else "unresolved"
                    )
            elif call_kind == "self":
                method_name = call_name.rsplit(".", 1)[-1]
                class_name = caller.parent_class
                candidates = [
                    item
                    for item in local_definitions
                    if item.kind == "method"
                    and item.parent_class == class_name
                    and item.name == method_name
                ]
                confidence = (
                    "resolved" if len(candidates) == 1
                    else "ambiguous" if candidates
                    else "unresolved"
                )
            else:
                confidence = "unresolved"
            edges.append(_edge(caller, call_node, call_name, confidence, candidates))
    return edges


def extract_call_edges(
    file_paths: Iterable[str | Path], repo_root: str | Path | None = None
) -> list[dict[str, Any]]:
    """Extract call edges from ``file_paths`` without guessing dynamic targets."""
    paths = [Path(path).resolve() for path in file_paths]
    root_path = Path(repo_root).resolve() if repo_root is not None else None
    if root_path is not None:
        paths = list(dict.fromkeys(paths + list(root_path.rglob("*.py"))))
    parsed: list[tuple[Path, bytes, Node]] = []
    all_definitions: list[_Definition] = []
    for path in paths:
        source = path.read_bytes()
        root = _PARSER.parse(source).root_node
        parsed.append((path, source, root))
        all_definitions.extend(_definitions(root, source, str(path), ()))
    known_files = {str(path) for path in paths}
    requested = {Path(path).resolve() for path in file_paths}
    return [
        edge
        for path, source, root in parsed
        if path in requested
        for edge in _extract_edges(path, source, root, all_definitions, known_files, root_path)
    ]


def extract_calls(
    file_path: str | Path, repo_root: str | Path | None = None
) -> list[dict[str, Any]]:
    """Extract call edges for one file."""
    return extract_call_edges([file_path], repo_root)


__all__ = ["extract_call_edges", "extract_calls"]
