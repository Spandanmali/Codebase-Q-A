"""Run retrieval evaluation against the repository-grounded benchmark set.

Usage:
    python -m src.evaluation.run_eval --config ast_vector
    python -m src.evaluation.run_eval --config ast_bm25
    python -m src.evaluation.run_eval --config ast_hybrid
    python -m src.evaluation.run_eval --config ast_hybrid_rerank
    python -m src.evaluation.run_eval --config ast_hybrid_rerank_graph
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

try:
    from config import load_settings
    from indexing.embedder import Embedder
    from indexing.pipeline import index_repo
    from indexing.store import VectorStore
    from retrieval.bm25 import BM25Index
    from retrieval.pipeline import RetrievalPipeline
    from retrieval.rerank import Reranker
except ModuleNotFoundError:
    from config import load_settings

    from ..indexing.embedder import Embedder
    from ..indexing.pipeline import index_repo
    from ..indexing.store import VectorStore
    from ..retrieval.bm25 import BM25Index
    from ..retrieval.pipeline import RetrievalPipeline
    from ..retrieval.rerank import Reranker

from .benchmark_schema import BenchmarkItem, load_benchmarks
from .metrics import hit_rate, mrr, recall_at_k


@dataclass(frozen=True)
class RetrievalConfig:
    """Named retrieval settings used by the evaluation registry."""

    chunker: str
    strategy: str = "vector"  # Options: "vector", "bm25", "hybrid"
    top_k: int = 5
    vector: bool | None = None
    bm25: bool | None = None
    hybrid: bool | None = None
    rerank: bool = False
    graph_expansion: bool = False

    def pipeline_flags(self) -> dict[str, bool]:
        """Return stage flags, preserving compatibility with strategy-only configs."""
        if self.vector is None and self.bm25 is None and self.hybrid is None:
            return {
                "vector": self.strategy in {"vector", "hybrid"},
                "bm25": self.strategy in {"bm25", "hybrid"},
                "hybrid": self.strategy == "hybrid",
                "rerank": self.rerank,
                "graph_expansion": self.graph_expansion,
            }
        return {
            "vector": self.vector is True,
            "bm25": self.bm25 is True,
            "hybrid": self.hybrid is True,
            "rerank": self.rerank,
            "graph_expansion": self.graph_expansion,
        }


CONFIG_REGISTRY: dict[str, RetrievalConfig] = {
    "fixed_vector": RetrievalConfig(chunker="fixed", strategy="vector"),
    "ast_vector": RetrievalConfig(chunker="ast", strategy="vector"),
    "ast_bm25": RetrievalConfig(chunker="ast", strategy="bm25"),
    "ast_hybrid": RetrievalConfig(chunker="ast", strategy="hybrid"),
    "ast_hybrid_rerank": RetrievalConfig(
        chunker="ast",
        strategy="hybrid",
        vector=True,
        bm25=True,
        hybrid=True,
        rerank=True,
    ),
    "ast_hybrid_rerank_graph": RetrievalConfig(
        chunker="ast",
        strategy="hybrid",
        vector=True,
        bm25=True,
        hybrid=True,
        rerank=True,
        graph_expansion=True,
    ),
}


@dataclass(frozen=True)
class _ScoredItem:
    item: BenchmarkItem
    retrieved: list[dict[str, object]]


def _get_chunk_dict_by_id(store: VectorStore, cid: str) -> dict[str, object] | None:
    """Safely retrieves a chunk dict from Chroma vector store using chunk ID."""
    try:
        res = store.collection.get(ids=[cid], include=["metadatas", "documents"])
        if res and res.get("ids") and len(res["ids"]) > 0:
            metadata = res["metadatas"][0] if res.get("metadatas") else {}
            document = res["documents"][0] if res.get("documents") else ""
            return {
                "id": cid,
                "document": document,
                "metadata": metadata,
            }
    except Exception:
        pass
    return None


def run_evaluation(
    config_name: str,
    *,
    benchmark_directory: str | Path | None = None,
    results_directory: str | Path = "results",
    configs: dict[str, RetrievalConfig] | None = None,
    benchmark_loader: Callable[..., list[BenchmarkItem]] = load_benchmarks,
) -> dict[str, object]:
    """Index pinned benchmark repositories, score retrieval, and save one result file."""
    registry = configs or CONFIG_REGISTRY
    try:
        retrieval_config = registry[config_name]
    except KeyError as error:
        available = ", ".join(sorted(registry))
        raise ValueError(f"unknown config {config_name!r}; choose one of: {available}") from error

    settings = load_settings()
    benchmarks = benchmark_loader(benchmark_directory, validate_repositories=False)
    embedder = Embedder(settings=settings)
    scored: list[_ScoredItem] = []
    indexed_repositories: set[tuple[str, str]] = set()

    for item in benchmarks:
        source = _repository_source(item.repo)
        repository_id = _repository_id(item.repo)
        index_key = (repository_id, item.commit.lower())

        if index_key not in indexed_repositories:
            index_repo(
                source,
                chunker=retrieval_config.chunker,
                commit=item.commit,
                allow_local=isinstance(source, Path),
            )
            indexed_repositories.add(index_key)

        store = VectorStore(
            settings.index_dir,
            repository_id,
            item.commit,
            retrieval_config.chunker,
            settings.embedding_model,
        )

        flags = retrieval_config.pipeline_flags()
        bm25_index = (
            BM25Index(
                settings.index_dir,
                repository_id,
                item.commit,
                retrieval_config.chunker,
            ).load()
            if flags["bm25"]
            else None
        )
        reranker = Reranker(settings=settings) if flags["rerank"] else None
        pipeline = RetrievalPipeline(
            vector_store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            reranker=reranker,
            chunk_loader=lambda cid: _get_chunk_dict_by_id(store, cid),
            **flags,
        )
        retrieved_chunks = pipeline.retrieve(item.question, retrieval_config.top_k)

        scored.append(_ScoredItem(item, retrieved_chunks))

    result = _build_result(
        config_name,
        retrieval_config,
        (settings.embedding_model, settings.generation_model),
        benchmarks,
        scored,
    )
    output_path = Path(results_directory) / f"{config_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_markdown(result)
    return result


def _build_result(
    config_name: str,
    retrieval_config: RetrievalConfig,
    model_names: tuple[str, str],
    benchmarks: Sequence[BenchmarkItem],
    scored: Sequence[_ScoredItem],
) -> dict[str, object]:
    answerable = [entry for entry in scored if entry.item.question_type != "unanswerable"]
    by_type: dict[str, list[_ScoredItem]] = defaultdict(list)
    for entry in answerable:
        by_type[entry.item.question_type].append(entry)

    metrics_by_type = {
        question_type: _aggregate(entries, retrieval_config.top_k)
        for question_type, entries in sorted(by_type.items())
    }
    metrics_by_type["overall"] = _aggregate(answerable, retrieval_config.top_k)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": config_name,
        "chunker": retrieval_config.chunker,
        "strategy": retrieval_config.strategy,
        "top_k": retrieval_config.top_k,
        "models": {"embedding": model_names[0], "generation": model_names[1]},
        "commit_hashes": sorted({entry.item.commit for entry in scored}),
        "metrics": metrics_by_type,
        "unanswerable": [
            {"id": entry.item.id, "question": entry.item.question}
            for entry in scored
            if entry.item.question_type == "unanswerable"
        ],
    }


def _aggregate(entries: Sequence[_ScoredItem], k: int) -> dict[str, object]:
    if not entries:
        return {"count": 0, "file": _empty_metrics(), "chunk": _empty_metrics()}

    return {
        "count": len(entries),
        "file": _average_metrics(
            [
                _item_metrics(entry.retrieved, entry.item, k, level="file")
                for entry in entries
            ]
        ),
        "chunk": _average_metrics(
            [
                _item_metrics(entry.retrieved, entry.item, k, level="chunk")
                for entry in entries
            ]
        ),
    }


def _item_metrics(
    retrieved: Sequence[dict[str, object]],
    item: BenchmarkItem,
    k: int,
    *,
    level: str,
) -> dict[str, float]:
    if level == "file":
        retrieved_items: Sequence[object] = [
            _metadata_value(result, "file_path") for result in retrieved
        ]
        gold_items: Sequence[object] = item.gold_files
    else:
        retrieved_items = [_metadata(result) for result in retrieved]
        gold_items = item.gold_ranges
    return {
        "recall_at_k": recall_at_k(retrieved_items, gold_items, k, level=level),
        "mrr": mrr(retrieved_items, gold_items, level=level),
        "hit_rate": hit_rate(retrieved_items, gold_items, k, level=level),
    }


def _average_metrics(values: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        name: sum(value[name] for value in values) / len(values)
        for name in ("recall_at_k", "mrr", "hit_rate")
    }


def _empty_metrics() -> dict[str, float]:
    return {"recall_at_k": 0.0, "mrr": 0.0, "hit_rate": 0.0}


def _metadata(result: dict[str, object]) -> dict[str, object]:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("retrieval result is missing metadata")
    return metadata


def _metadata_value(result: dict[str, object], key: str) -> str:
    value = _metadata(result).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"retrieval metadata field {key!r} must be a non-empty string")
    return value


def _repository_source(repo: str) -> Path | str:
    path = Path(repo)
    return path if path.is_dir() else repo


def _repository_id(repo: str) -> str:
    path = Path(repo)
    if path.is_dir():
        return str(path.resolve())
    parsed = urlsplit(repo)
    return parsed.path.removesuffix(".git").strip("/") or repo


def _print_markdown(result: dict[str, object]) -> None:
    print(f"## Retrieval evaluation: `{result['config']}` (Strategy: `{result['strategy']}`)")
    print()
    print("| Question type | Count | Level | Recall@k | MRR | Hit rate |")
    print("|---|---:|---|---:|---:|---:|")
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    for question_type, values in metrics.items():
        assert isinstance(values, dict)
        count = values["count"]
        for level in ("file", "chunk"):
            level_values = values[level]
            assert isinstance(level_values, dict)
            print(
                f"| {question_type} | {count} | {level} | "
                f"{level_values['recall_at_k']:.3f} | {level_values['mrr']:.3f} | "
                f"{level_values['hit_rate']:.3f} |"
            )
    unanswerable = result["unanswerable"]
    assert isinstance(unanswerable, list)
    print()
    print(f"Unanswerable items (Phase 8): {len(unanswerable)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repository retrieval configurations.")
    parser.add_argument("--config", required=True, choices=sorted(CONFIG_REGISTRY))
    args = parser.parse_args()
    run_evaluation(args.config)


if __name__ == "__main__":
    main()