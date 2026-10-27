import pytest

from src.chunking.models import Chunk
from src.evaluation.benchmark_schema import GoldRange
from src.evaluation.metrics import hit_rate, mrr, recall_at_k


def test_file_metrics_use_ranked_results() -> None:
    retrieved = ["wrong.py", "b.py", "a.py"]
    gold = ["a.py", "b.py"]

    assert recall_at_k(retrieved, gold, 2) == pytest.approx(0.5)
    assert mrr(retrieved, gold) == pytest.approx(0.5)
    assert hit_rate(retrieved, gold, 1) == 0.0
    assert hit_rate(retrieved, gold, 2) == 1.0


def test_chunk_metrics_match_same_file_when_ranges_overlap() -> None:
    retrieved = [
        Chunk("wrong", "a.py", 1, 4, ""),
        Chunk("overlap", "a.py", 10, 15, ""),
        Chunk("other", "a.py", 30, 35, ""),
    ]
    gold = [
        GoldRange(file="a.py", start=15, end=20),
        GoldRange(file="a.py", start=30, end=32),
    ]

    assert recall_at_k(retrieved, gold, 2, level="chunk") == pytest.approx(0.5)
    assert mrr(retrieved, gold, level="chunk") == pytest.approx(0.5)
    assert hit_rate(retrieved, gold, 1, level="chunk") == 0.0
    assert hit_rate(retrieved, gold, 2, level="chunk") == 1.0


def test_chunk_metrics_do_not_match_ranges_from_another_file() -> None:
    retrieved = [Chunk("same-lines", "b.py", 10, 20, "")]
    gold = [GoldRange(file="a.py", start=10, end=20)]

    assert recall_at_k(retrieved, gold, 1, level="chunk") == 0.0
    assert mrr(retrieved, gold, level="chunk") == 0.0
    assert hit_rate(retrieved, gold, 1, level="chunk") == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, hit_rate])
def test_top_k_requires_positive_k(metric) -> None:
    with pytest.raises(ValueError, match="positive"):
        if metric is recall_at_k:
            metric(["a.py"], ["a.py"], 0)
        else:
            metric(["a.py"], ["a.py"], 0)
