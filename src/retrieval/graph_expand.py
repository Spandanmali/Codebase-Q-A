"""Graph-aware expansion of retrieved source chunks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    from chunking.models import Chunk
except ModuleNotFoundError:
    from ..chunking.models import Chunk

ChunkLoader = Callable[[str], Chunk | None]


def _chunk_id(chunk: Chunk) -> str:
    if not isinstance(chunk.id, str) or not chunk.id:
        raise ValueError("seed chunks must have non-empty string ids")
    return chunk.id


def _chunk_from_graph(graph: Any, chunk_id: str) -> Chunk | None:
    chunks = graph.graph.get("chunks", {})
    if isinstance(chunks, Mapping):
        chunk = chunks.get(chunk_id)
        if isinstance(chunk, Chunk):
            return chunk

    for _, attributes in graph.nodes(data=True):
        node_chunks = attributes.get("chunks", {})
        if isinstance(node_chunks, Mapping):
            chunk = node_chunks.get(chunk_id)
            if isinstance(chunk, Chunk):
                return chunk
        if isinstance(node_chunks, Sequence) and not isinstance(node_chunks, (str, bytes)):
            for chunk in node_chunks:
                if isinstance(chunk, Chunk) and chunk.id == chunk_id:
                    return chunk
    return None


def _edge_is_allowed(attributes: Mapping[str, Any], resolved_only: bool) -> bool:
    if resolved_only and attributes.get("resolved", True) is False:
        return False
    return attributes.get("type") in {"calls", "imports"}


def _related_nodes(graph: Any, node: str, resolved_only: bool) -> set[str]:
    related: set[str] = set()
    for source, _, attributes in graph.in_edges(node, data=True):
        if attributes.get("type") == "calls" and _edge_is_allowed(
            attributes, resolved_only
        ):
            related.add(source)
    for _, target, attributes in graph.out_edges(node, data=True):
        if not _edge_is_allowed(attributes, resolved_only):
            continue
        related.add(target)
    return related


def _token_count(chunk: Chunk) -> int:
    return len(chunk.text.split())


def expand(
    seed_chunks: Sequence[Chunk],
    graph: Any,
    depth: int = 1,
    max_added: int = 10,
    *,
    query: str = "",
    reranker: Any,
    token_budget: int = 4096,
    resolved_only: bool = True,
    chunk_loader: ChunkLoader | None = None,
) -> list[Chunk]:
    """Add graph-related chunks, rerank all candidates, and enforce a token budget.

    ``max_added`` limits graph-derived chunks; seeds are always considered
    candidates. The final result is the top ``len(seed_chunks) + max_added``
    reranked candidates that fit in ``token_budget``. Graph nodes without a
    corresponding chunk object are ignored.
    """
    if depth < 0 or depth > 2:
        raise ValueError("depth must be between 0 and 2")
    if max_added < 0:
        raise ValueError("max_added must not be negative")
    if token_budget < 0:
        raise ValueError("token_budget must not be negative")
    if reranker is None or not callable(getattr(reranker, "rerank", None)):
        raise ValueError("reranker with a rerank method is required")

    seeds_by_id = {_chunk_id(chunk): chunk for chunk in seed_chunks}
    candidates = dict(seeds_by_id)
    if depth and max_added:
        frontier: set[str] = set()
        for chunk_id in seeds_by_id:
            if graph.has_node(chunk_id):
                frontier.add(chunk_id)
            else:
                frontier.update(
                    node
                    for node, attributes in graph.nodes(data=True)
                    if chunk_id in attributes.get("chunk_ids", [])
                )

        visited = set(frontier)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(_related_nodes(graph, node, resolved_only) - visited)
            visited.update(next_frontier)
            frontier = next_frontier
            for node in frontier:
                for chunk_id in graph.nodes[node].get("chunk_ids", []):
                    if chunk_id in candidates or len(candidates) - len(seeds_by_id) >= max_added:
                        continue
                    chunk = (
                        chunk_loader(chunk_id)
                        if chunk_loader is not None
                        else _chunk_from_graph(graph, chunk_id)
                    )
                    if chunk is not None and chunk.id == chunk_id:
                        candidates[chunk_id] = chunk

    ranked = reranker.rerank(query, list(candidates.values()), len(candidates))
    by_id = {chunk.id: chunk for chunk in candidates.values()}
    ordered = [by_id[chunk.id] for chunk in ranked if chunk.id in by_id]

    result: list[Chunk] = []
    total_tokens = 0
    limit = len(seeds_by_id) + max_added
    for chunk in ordered:
        if len(result) >= limit:
            break
        tokens = _token_count(chunk)
        if total_tokens + tokens > token_budget:
            continue
        result.append(chunk)
        total_tokens += tokens
    return result


__all__ = ["expand"]
