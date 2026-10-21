"""Command-line entry points for the application."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from generation.answer import answer
    from indexing.pipeline import index_repo
except ModuleNotFoundError:
    from ..generation.answer import answer
    from ..indexing.pipeline import index_repo


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.app.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="index a GitHub repository")
    index_parser.add_argument("source", help="GitHub URL, or local path with --local")
    index_parser.add_argument("--chunker", choices=("ast", "fixed"), default="ast")
    index_parser.add_argument("--commit")
    index_parser.add_argument("--force", action="store_true")
    index_parser.add_argument("--local", action="store_true")
    ask_parser = subparsers.add_parser("ask", help="ask a question about an indexed repository")
    ask_parser.add_argument("repo_id")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "ask":
        response, chunks = answer(args.question, args.repo_id, args.k)
        print(response)
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            print(
                f"[{metadata.get('file_path', 'unknown')}:"
                f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}]"
            )
        return 0

    if args.local:
        source: str | Path = Path(args.source)
    else:
        source = args.source
    stats = index_repo(
        source,
        chunker=args.chunker,
        commit=args.commit,
        force=args.force,
        allow_local=args.local,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
