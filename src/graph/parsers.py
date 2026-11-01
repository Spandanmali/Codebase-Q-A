"""Tree-sitter parsers selected by source-file extension."""

from pathlib import Path

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Parser

_PARSERS = {
    ".js": Parser(Language(tree_sitter_javascript.language())),
    ".jsx": Parser(Language(tree_sitter_javascript.language())),
    ".mjs": Parser(Language(tree_sitter_javascript.language())),
    ".cjs": Parser(Language(tree_sitter_javascript.language())),
    ".ts": Parser(Language(tree_sitter_typescript.language_typescript())),
    ".tsx": Parser(Language(tree_sitter_typescript.language_tsx())),
}


def get_parser(file_path: str | Path) -> Parser | None:
    """Return the parser for a supported JavaScript/TypeScript extension."""
    return _PARSERS.get(Path(file_path).suffix.lower())


parser_for_path = get_parser


__all__ = ["get_parser", "parser_for_path"]
