"""Extract Python definitions and imports with tree-sitter."""

from pathlib import Path
from typing import Any

import tree_sitter_python
from tree_sitter import Language, Node, Parser

_PARSER = Parser(Language(tree_sitter_python.language()))


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _line_range(node: Node) -> dict[str, int]:
    return {
        "start_line": node.start_point.row + 1,
        "end_line": node.end_point.row + 1,
    }


def _definition_node(node: Node) -> Node | None:
    if node.type in {"function_definition", "class_definition"}:
        return node
    if node.type == "decorated_definition":
        for child in node.named_children:
            if child.type in {"function_definition", "class_definition"}:
                return child
    return None


def _definitions(
    node: Node,
    source: bytes,
    class_stack: tuple[str, ...] = (),
    inside_function: bool = False,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    definition = _definition_node(node)
    if definition is not None:
        name_node = definition.child_by_field_name("name")
        name = _text(name_node, source) if name_node else ""
        if definition.type == "class_definition":
            kind = "class"
        elif class_stack and not inside_function:
            kind = "method"
        else:
            kind = "function"
        entry: dict[str, Any] = {"name": name, "type": kind, **_line_range(node)}
        if class_stack and kind == "method":
            entry["parent_class"] = class_stack[-1]
        result.append(entry)

        next_classes = class_stack + ((name,) if definition.type == "class_definition" else ())
        body = definition.child_by_field_name("body")
        if body:
            for child in body.named_children:
                result.extend(
                    _definitions(
                        child,
                        source,
                        next_classes,
                        inside_function or definition.type == "function_definition",
                    )
                )
        return result

    for child in node.named_children:
        result.extend(_definitions(child, source, class_stack, inside_function))
    return result


def _import_names(node: Node, source: bytes) -> list[dict[str, str | None]]:
    names: list[dict[str, str | None]] = []
    for child in node.named_children:
        if child.type in {"dotted_name", "aliased_import"}:
            value = _text(child, source)
            if child.type == "aliased_import":
                imported, _, alias = value.partition(" as ")
                names.append({"name": imported, "alias": alias or None})
            else:
                names.append({"name": value, "alias": None})
    return names


def _module_name(path: Path, repo_root: Path) -> tuple[str, ...]:
    relative = path.resolve().relative_to(repo_root.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return tuple(parts)


def _candidate_path(module: tuple[str, ...], repo_root: Path) -> str | None:
    if not module:
        candidate = repo_root / "__init__.py"
        return str(candidate) if candidate.is_file() else None
    base = repo_root.joinpath(*module)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return str(candidate)
    return None


def _resolved_import(
    module: str,
    level: int,
    imported_names: list[dict[str, str | None]],
    file_path: Path,
    repo_root: Path | None,
) -> str | None:
    if repo_root is None or level == 0:
        return None
    try:
        current = list(_module_name(file_path, repo_root))
    except ValueError:
        return None
    package = current if file_path.name == "__init__.py" else current[:-1]
    if level > len(package) + 1:
        return None
    base = package[: len(package) - level + 1]
    if module:
        base.extend(module.split("."))
    if not module and len(imported_names) == 1:
        sibling = tuple(base) + tuple(imported_names[0]["name"].split("."))
        sibling_path = _candidate_path(sibling, repo_root)
        if sibling_path:
            return sibling_path
    resolved = _candidate_path(tuple(base), repo_root)
    if resolved:
        return resolved
    return None


def _imports(
    root: Node, source: bytes, file_path: Path, repo_root: Path | None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    nodes = []
    stack = list(root.named_children)
    while stack:
        node = stack.pop()
        if node.type in {"import_statement", "import_from_statement"}:
            nodes.append(node)
        stack.extend(reversed(node.named_children))
    for node in sorted(nodes, key=lambda item: item.start_byte):
        if node.type == "import_statement":
            result.append(
                {
                    "kind": "import",
                    "module": None,
                    "level": 0,
                    "names": _import_names(node, source),
                    **_line_range(node),
                    "resolved_path": None,
                }
            )
        elif node.type == "import_from_statement":
            relative_node = next(
                (child for child in node.named_children if child.type == "relative_import"), None
            )
            if relative_node:
                relative_text = _text(relative_node, source)
                level = len(relative_text) - len(relative_text.lstrip("."))
                module = relative_text[level:]
                name_nodes = node.named_children[1:]
            else:
                level = 0
                module = _text(node.named_children[0], source)
                name_nodes = node.named_children[1:]
            names = []
            for child in name_nodes:
                if child.type == "wildcard_import":
                    names.append({"name": "*", "alias": None})
                elif child.type in {"dotted_name", "aliased_import"}:
                    value = _text(child, source)
                    if child.type == "aliased_import":
                        imported, _, alias = value.partition(" as ")
                        names.append({"name": imported, "alias": alias or None})
                    else:
                        names.append({"name": value, "alias": None})
            result.append(
                {
                    "kind": "from",
                    "module": module.lstrip("."),
                    "level": level,
                    "names": names,
                    **_line_range(node),
                    "resolved_path": _resolved_import(
                        module.lstrip("."), level, names, file_path, repo_root
                    ),
                }
            )
    return result


def extract_file(file_path: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Extract definitions and imports from one Python file.

    Paths in ``resolved_path`` are absolute strings. Relative imports are
    resolved from the source file's package, checking ``module.py`` and
    ``module/__init__.py``. Missing or external modules have a null path.
    """
    path = Path(file_path)
    source = path.read_bytes()
    tree = _PARSER.parse(source)
    root = tree.root_node
    root_path = Path(repo_root).resolve() if repo_root is not None else None
    return {
        "file_path": str(path),
        "definitions": _definitions(root, source),
        "imports": _imports(root, source, path, root_path),
    }


def extract_files(
    file_paths: list[str | Path], repo_root: str | Path | None = None
) -> list[dict[str, Any]]:
    """Extract structured graph data for each file in ``file_paths``."""
    return [extract_file(path, repo_root) for path in file_paths]


__all__ = ["extract_file", "extract_files"]
