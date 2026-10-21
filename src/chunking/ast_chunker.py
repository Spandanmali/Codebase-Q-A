import logging
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from .fixed import chunk_fixed
from .models import Chunk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    symbol_name: str | None
    symbol_type: str
    parent_class: str | None = None
    docstring: str | None = None
    split_lines: tuple[int, ...] = ()
    context: str | None = None


def _definition(node: Node) -> Node:
    if node.type == "decorated_definition":
        for child in node.named_children:
            if child.type in {"function_definition", "class_definition"}:
                return child
    return node


def _name(node: Node, source: bytes) -> str | None:
    name = node.child_by_field_name("name")
    return source[name.start_byte:name.end_byte].decode("utf-8") if name else None


def _docstring(node: Node, source: bytes) -> tuple[int, str] | None:
    body = node.child_by_field_name("body")
    if not body or not body.named_children:
        return None
    first = body.named_children[0]
    if first.type != "expression_statement" or not first.named_children:
        return None
    expression = first.named_children[0]
    if expression.type != "string":
        return None
    return first.end_point.row + 1, source[expression.start_byte:expression.end_byte].decode(
        "utf-8"
    )


def _line_span(node: Node, start: int | None = None, end: int | None = None) -> tuple[int, int]:
    return (
        start if start is not None else node.start_point.row + 1,
        end if end is not None else node.end_point.row + 1,
    )


def _statement_starts(node: Node) -> tuple[int, ...]:
    body = node.child_by_field_name("body")
    if not body:
        return ()
    return tuple(child.start_point.row + 1 for child in body.named_children)


def _signature(node: Node, source: bytes) -> str:
    definition = _definition(node)
    body = definition.child_by_field_name("body")
    if not body:
        return ""
    start_line = definition.start_point.row
    end_line = body.start_point.row
    return b"".join(source.splitlines(keepends=True)[start_line:end_line]).decode(
        "utf-8"
    ).rstrip()


def _method_span(node: Node, source: bytes, parent_class: str) -> _Span:
    definition = _definition(node)
    start, end = _line_span(node)
    return _Span(
        start,
        end,
        _name(definition, source),
        "method",
        parent_class,
        _docstring(definition, source)[1] if _docstring(definition, source) else None,
        _statement_starts(definition),
        _signature(node, source),
    )


def _class_spans(node: Node, source: bytes, parent_class: str | None = None) -> list[_Span]:
    definition = _definition(node)
    class_name = _name(definition, source)
    start, end = _line_span(node)
    body = definition.child_by_field_name("body")
    children = list(body.named_children) if body else []
    if not children:
        return [
            _Span(
                start,
                end,
                class_name,
                "class",
                parent_class=parent_class,
                split_lines=_statement_starts(definition),
                context=_signature(node, source),
            ),
        ]

    spans: list[_Span] = []
    first_child_start = children[0].start_point.row + 1
    docstring = _docstring(definition, source)
    header_end = docstring[0] if docstring else max(start, first_child_start - 1)
    spans.append(
        _Span(
            start,
            header_end,
            class_name,
            "class",
            parent_class,
            docstring[1] if docstring else None,
            _statement_starts(definition),
            _signature(node, source),
        )
    )

    for child in children:
        child_definition = _definition(child)
        if (
            child.type in {"function_definition", "decorated_definition"}
            and child_definition.type == "function_definition"
        ):
            spans.append(_method_span(child, source, class_name or parent_class or ""))
        elif (
            child.type in {"class_definition", "decorated_definition"}
            and child_definition.type == "class_definition"
        ):
            spans.extend(_class_spans(child, source, class_name))

    return _fill_gaps(spans, start, end, "class_body", class_name)


def _fill_gaps(
    spans: list[_Span],
    start: int,
    end: int,
    gap_type: str,
    parent_class: str | None = None,
) -> list[_Span]:
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    result: list[_Span] = []
    cursor = start
    for span in ordered:
        if span.start > cursor:
            result.append(_Span(cursor, span.start - 1, None, gap_type, parent_class))
        if span.end >= cursor:
            result.append(span)
            cursor = span.end + 1
    if cursor <= end:
        result.append(_Span(cursor, end, None, gap_type, parent_class))
    return result


def _python_spans(root: Node, source: bytes, line_count: int) -> list[_Span]:
    spans: list[_Span] = []
    for node in root.named_children:
        definition = _definition(node)
        if definition.type == "function_definition":
            docstring = _docstring(definition, source)
            start, end = _line_span(node)
            spans.append(
                _Span(
                    start,
                    end,
                    _name(definition, source),
                    "function",
                    docstring=docstring[1] if docstring else None,
                    split_lines=_statement_starts(definition),
                    context=_signature(node, source),
                )
            )
        elif definition.type == "class_definition":
            spans.extend(_class_spans(node, source))
        else:
            start, end = _line_span(node)
            spans.append(_Span(start, end, None, "module"))
    return _fill_gaps(spans, 1, line_count, "module")


def _split_lines(
    start: int,
    end: int,
    lines: list[str],
    boundaries: tuple[int, ...],
) -> list[tuple[int, int]]:
    points = sorted({start, *(line for line in boundaries if start < line <= end), end + 1})
    parts: list[tuple[int, int]] = []
    pending_start: int | None = None
    pending_end: int | None = None
    for part_start, next_start in zip(points, points[1:]):
        part_end = next_start - 1
        part_text = "".join(lines[part_start - 1 : part_end])
        if len(part_text) / 4 <= 512:
            combined_start = pending_start if pending_start is not None else part_start
            combined_end = part_end
            if len("".join(lines[combined_start - 1 : combined_end])) / 4 <= 512:
                pending_start = combined_start
                pending_end = combined_end
                continue
            if pending_start is not None and pending_end is not None:
                parts.append((pending_start, pending_end))
            pending_start, pending_end = part_start, part_end
            continue

        if pending_start is not None and pending_end is not None:
            parts.append((pending_start, pending_end))
            pending_start = pending_end = None
        while part_start <= part_end:
            end_line = part_end
            while (
                end_line >= part_start
                and len("".join(lines[part_start - 1 : end_line])) / 4 > 512
            ):
                end_line -= 1
            if end_line < part_start:
                end_line = part_start
            parts.append((part_start, end_line))
            part_start = end_line + 1
    if pending_start is not None and pending_end is not None:
        parts.append((pending_start, pending_end))
    return parts


def _post_process(
    chunks: list[Chunk],
    spans: list[_Span],
    lines: list[str],
) -> list[Chunk]:
    result: list[Chunk] = []
    for chunk, span in zip(chunks, spans):
        if len(chunk.text) / 4 <= 512:
            result.append(chunk)
            continue
        parts = _split_lines(span.start, span.end, lines, span.split_lines)
        for index, (start, end) in enumerate(parts, start=1):
            result.append(
                Chunk(
                    id=f"{chunk.id}:part{index}",
                    file_path=chunk.file_path,
                    start_line=start,
                    end_line=end,
                    text="".join(lines[start - 1 : end]),
                    language=chunk.language,
                    chunker=chunk.chunker,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    parent_class=chunk.parent_class,
                    docstring=chunk.docstring,
                    context=span.context,
                )
            )
    return result


def chunk_python(file_path: str | Path, content: str) -> list[Chunk]:
    """Split valid Python source into non-overlapping AST-aware chunks."""
    path = str(file_path)
    source = content.encode("utf-8")
    lines = content.splitlines(keepends=True)
    if not lines:
        LOGGER.warning("Python file %s is empty; using fixed chunking", path)
        return chunk_fixed(path, content)

    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(source)
    if tree.root_node.has_error:
        LOGGER.warning("Could not parse Python file %s; using fixed chunking", path)
        return chunk_fixed(path, content)

    spans = _python_spans(tree.root_node, source, len(lines))
    usable = [
        span
        for span in spans
        if any(line.strip() for line in lines[span.start - 1 : span.end])
    ]
    if not usable:
        LOGGER.warning("Python file %s has no usable AST chunks; using fixed chunking", path)
        return chunk_fixed(path, content)

    chunks = [
        Chunk(
            id=f"ast:{path}:{span.start}-{span.end}",
            file_path=path,
            start_line=span.start,
            end_line=span.end,
            text="".join(lines[span.start - 1 : span.end]),
            language="python",
            chunker="ast",
            symbol_name=span.symbol_name,
            symbol_type=span.symbol_type,
            parent_class=span.parent_class,
            docstring=span.docstring,
        )
        for span in spans
        if any(line.strip() for line in lines[span.start - 1 : span.end])
    ]
    return _post_process(chunks, usable, lines)


def chunk_file(file_path: str | Path, content: str) -> list[Chunk]:
    """Use AST chunking for Python and fixed chunking for supported documents."""
    if Path(file_path).suffix.lower() == ".py":
        return chunk_python(file_path, content)
    return chunk_fixed(file_path, content)
