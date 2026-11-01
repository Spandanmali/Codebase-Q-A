"""Extract Python and JavaScript/TypeScript definitions and imports."""

import ast
import logging
from pathlib import Path
from typing import Any

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from ingestion.languages import EXTENSION_TO_LANGUAGE
from .parsers import parser_for_path

_PARSER = Parser(Language(tree_sitter_python.language()))
_LOGGER = logging.getLogger(__name__)
_JS_RESOLUTION_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


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


def _js_string(node: Node | None, source: bytes) -> str | None:
    if node is None or node.type not in {"string", "string_fragment"}:
        return None
    value = _text(node, source)
    if node.type == "string_fragment":
        return value
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, str) else None


def _js_name(node: Node | None, source: bytes) -> str:
    return _text(node, source) if node is not None else ""


def _js_definition(
    node: Node,
    source: bytes,
    name: str,
    kind: str,
    class_stack: tuple[str, ...],
    function_stack: tuple[str, ...],
    *,
    parent_class: str | None = None,
    parent_function: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "type": kind,
        **_line_range(node),
    }
    if parent_class is not None:
        entry["parent_class"] = parent_class
    if parent_function is not None:
        entry["parent_function"] = parent_function
    return entry


def _js_definitions(
    node: Node,
    source: bytes,
    class_stack: tuple[str, ...] = (),
    function_stack: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if node.type == "class_declaration":
        name = _js_name(node.child_by_field_name("name"), source) or "default"
        result.append(
            _js_definition(
                node,
                source,
                name,
                "class",
                class_stack,
                function_stack,
                parent_class=class_stack[-1] if class_stack else None,
                parent_function=function_stack[-1] if function_stack else None,
            )
        )
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.named_children:
                result.extend(_js_definitions(child, source, (*class_stack, name), function_stack))
        return result

    if node.type == "method_definition":
        name = _js_name(node.child_by_field_name("name"), source)
        if name:
            qualified_parent = ".".join((*class_stack, *function_stack)) or None
            result.append(
                _js_definition(
                    node,
                    source,
                    name,
                    "method",
                    class_stack,
                    function_stack,
                    parent_class=class_stack[-1] if class_stack else None,
                    parent_function=qualified_parent if function_stack else None,
                )
            )
            body = node.child_by_field_name("body")
            if body is not None:
                for child in body.named_children:
                    result.extend(
                        _js_definitions(
                            child,
                            source,
                            class_stack,
                            (*function_stack, name),
                        )
                    )
        return result

    if node.type in {"function_declaration", "function_expression", "arrow_function"}:
        name = _js_name(node.child_by_field_name("name"), source)
        if name:
            kind = "function"
            parent_class = class_stack[-1] if class_stack else None
            parent_function = ".".join((*class_stack, *function_stack)) or None
            result.append(
                _js_definition(
                    node,
                    source,
                    name,
                    kind,
                    class_stack,
                    function_stack,
                    parent_class=parent_class,
                    parent_function=parent_function if function_stack else None,
                )
            )
        body = node.child_by_field_name("body")
        if body is not None:
            next_stack = (*function_stack, name) if name else function_stack
            for child in body.named_children:
                result.extend(_js_definitions(child, source, class_stack, next_stack))
        return result

    if node.type == "variable_declarator":
        name = _js_name(node.child_by_field_name("name"), source)
        value = node.child_by_field_name("value")
        if name and value is not None and value.type in {"arrow_function", "function_expression"}:
            result.append(
                _js_definition(
                    node,
                    source,
                    name,
                    "function",
                    class_stack,
                    function_stack,
                    parent_class=class_stack[-1] if class_stack else None,
                    parent_function=".".join((*class_stack, *function_stack)) or None,
                )
            )
            body = value.child_by_field_name("body")
            if body is not None:
                for child in body.named_children:
                    result.extend(
                        _js_definitions(child, source, class_stack, (*function_stack, name))
                    )
            return result

    if node.type == "export_statement":
        declaration = node.child_by_field_name("declaration") or node.child_by_field_name("value")
        if declaration is not None and declaration.type in {
            "class_declaration",
            "function_expression",
            "arrow_function",
        }:
            if (
                declaration.type != "class_declaration"
                and not declaration.child_by_field_name("name")
            ):
                synthetic = _js_definition(
                    declaration,
                    source,
                    "default",
                    "function",
                    class_stack,
                    function_stack,
                )
                result.append(synthetic)
                body = declaration.child_by_field_name("body")
                if body is not None:
                    for child in body.named_children:
                        result.extend(_js_definitions(child, source, class_stack, ("default",)))
                return result
            return _js_definitions(declaration, source, class_stack, function_stack)

    for child in node.named_children:
        result.extend(_js_definitions(child, source, class_stack, function_stack))
    return result


def _js_specifier(node: Node, source: bytes) -> str | None:
    return _js_string(node, source)


def _js_import_names(clause: Node | None, source: bytes) -> list[dict[str, str | None]]:
    names: list[dict[str, str | None]] = []
    if clause is None:
        return names
    for child in clause.named_children:
        if child.type == "identifier":
            names.append({"name": "default", "alias": _text(child, source)})
        elif child.type == "namespace_import":
            alias = child.named_children[-1] if child.named_children else None
            names.append({"name": "*", "alias": _js_name(alias, source) or None})
        elif child.type == "named_imports":
            for item in child.named_children:
                if item.type != "import_specifier":
                    continue
                imported = _js_name(item.child_by_field_name("name"), source)
                alias = _js_name(item.child_by_field_name("alias"), source) or None
                names.append({"name": imported, "alias": alias})
    return names


def _js_resolve_import(specifier: str, file_path: Path, repo_root: Path | None) -> str | None:
    if repo_root is None or not specifier.startswith(("./", "../")):
        return None
    root = repo_root.resolve()
    base = (file_path.resolve().parent / specifier).resolve()
    try:
        base.relative_to(root)
    except ValueError:
        return None
    suffix = base.suffix.lower()
    stem = base.with_suffix("") if suffix in _JS_RESOLUTION_EXTENSIONS else base
    candidates = [stem.with_suffix(extension) for extension in _JS_RESOLUTION_EXTENSIONS]
    candidates.extend(stem / f"index{extension}" for extension in _JS_RESOLUTION_EXTENSIONS)
    for candidate in candidates:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return str(candidate)
    return None


def _js_import(
    node: Node, source: bytes, file_path: Path, repo_root: Path | None
) -> dict[str, Any] | None:
    if node.type == "import_statement":
        specifier = _js_specifier(node.child_by_field_name("source"), source)
        if specifier is None:
            return {"kind": "import", "unresolved": True, **_line_range(node)}
        clause = next(
            (child for child in node.named_children if child.type == "import_clause"),
            None,
        )
        return {
            "kind": "import",
            "module": specifier,
            "names": _js_import_names(clause, source),
            "is_type": bool(clause and "type" in _text(node, source).split()),
            **_line_range(node),
            "resolved_path": _js_resolve_import(specifier, file_path, repo_root),
            "external": not specifier.startswith(("./", "../")),
            "unresolved": specifier.startswith(("./", "../"))
            and _js_resolve_import(specifier, file_path, repo_root) is None,
        }
    if node.type == "export_statement":
        specifier = _js_specifier(node.child_by_field_name("source"), source)
        if specifier is None:
            return None
        return {
            "kind": "export",
            "module": specifier,
            "names": [],
            **_line_range(node),
            "resolved_path": _js_resolve_import(specifier, file_path, repo_root),
            "external": not specifier.startswith(("./", "../")),
            "unresolved": specifier.startswith(("./", "../"))
            and _js_resolve_import(specifier, file_path, repo_root) is None,
        }
    if node.type == "call_expression":
        function = node.child_by_field_name("function")
        if function is None:
            return None
        function_name = _text(function, source)
        if function_name == "import":
            return {
                "kind": "import",
                "module": None,
                "names": [],
                **_line_range(node),
                "resolved_path": None,
                "unresolved": True,
            }
        if function_name != "require":
            return None
        arguments = node.child_by_field_name("arguments")
        argument = arguments.named_children[0] if arguments and arguments.named_children else None
        specifier = _js_specifier(argument, source)
        if specifier is None:
            return {
                "kind": "require",
                "module": None,
                "names": [],
                **_line_range(node),
                "resolved_path": None,
                "unresolved": True,
            }
        return {
            "kind": "require",
            "module": specifier,
            "names": [],
            **_line_range(node),
            "resolved_path": _js_resolve_import(specifier, file_path, repo_root),
            "external": not specifier.startswith(("./", "../")),
            "unresolved": specifier.startswith(("./", "../"))
            and _js_resolve_import(specifier, file_path, repo_root) is None,
        }
    return None


def _js_imports(
    root: Node, source: bytes, file_path: Path, repo_root: Path | None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in {"import_statement", "export_statement", "call_expression"}:
            item = _js_import(node, source, file_path, repo_root)
            if item is not None:
                result.append(item)
        stack.extend(reversed(node.named_children))
    return sorted(result, key=lambda item: (item["start_line"], item["end_line"]))


def _extract_javascript(
    path: Path, repo_root: Path | None, parser: Parser
) -> dict[str, Any]:
    source = path.read_bytes()
    tree = parser.parse(source)
    if tree.root_node.has_error:
        _LOGGER.warning("Skipping unparsable JavaScript/TypeScript file %s", path)
        return {
            "file_path": str(path),
            "definitions": [],
            "imports": [],
            "skipped": True,
        }
    return {
        "file_path": str(path),
        "definitions": _js_definitions(tree.root_node, source),
        "imports": _js_imports(tree.root_node, source, path, repo_root),
    }


def extract_file(file_path: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Extract definitions and imports from one supported source file.

    Paths in ``resolved_path`` are absolute strings. Relative imports are
    resolved from the source file's package, checking ``module.py`` and
    ``module/__init__.py``. Missing or external modules have a null path.
    """
    path = Path(file_path)
    language = EXTENSION_TO_LANGUAGE.get(path.suffix.lower())
    if language in {"javascript", "typescript"}:
        parser = parser_for_path(path)
        if parser is not None:
            root_path = Path(repo_root).resolve() if repo_root is not None else None
            return _extract_javascript(path, root_path, parser)
    if language != "python":
        return {"file_path": str(path), "definitions": [], "imports": []}
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
