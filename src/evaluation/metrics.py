"""Retrieval metrics for repository files and source-code chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

EvaluationLevel = Literal["file", "chunk"]


def recall_at_k(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    k: int,
    *,
    level: EvaluationLevel = "file",
) -> float:
    """Return the fraction of gold items covered by the first ``k`` results.

    At file level, two files match when their paths are equal.  At chunk level,
    a retrieved range matches a gold range when they refer to the same file and
    their inclusive line ranges overlap.  For example, with gold files
    ``["a.py", "b.py"]`` and the first two results ``["x.py", "b.py"]``,
    ``recall_at_k(..., 2)`` is ``1 / 2``.  With gold range ``a.py:10-20`` and
    retrieved range ``a.py:20-25``, chunk recall is ``1.0`` because line 20
    overlaps.

    An empty gold set returns ``0.0`` rather than a vacuous perfect score.
    """
    _validate_k(k)
    _validate_level(level)
    if not gold:
        return 0.0

    if level == "file":
        gold_items = {_file_path(item) for item in gold}
        retrieved_items = {_file_path(item) for item in retrieved[:k]}
        return len(gold_items & retrieved_items) / len(gold_items)

    gold_ranges = [_line_range(item) for item in gold]
    retrieved_ranges = [_line_range(item) for item in retrieved[:k]]
    covered = sum(
        any(_ranges_overlap(retrieved_range, gold_range) for retrieved_range in retrieved_ranges)
        for gold_range in gold_ranges
    )
    return covered / len(gold_ranges)


def mrr(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    *,
    level: EvaluationLevel = "file",
) -> float:
    """Return the reciprocal rank of the first relevant result.

    For example, if the ranked files are ``["wrong.py", "right.py"]`` and
    ``"right.py"`` is gold, MRR is ``1 / 2 = 0.5``.  At chunk level, the
    first chunk whose inclusive range overlaps a gold range determines the
    rank.  If no result is relevant, the metric is ``0.0``.
    """
    _validate_level(level)
    for rank, item in enumerate(retrieved, start=1):
        if _is_relevant(item, gold, level):
            return 1.0 / rank
    return 0.0


def hit_rate(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    k: int,
    *,
    level: EvaluationLevel = "file",
) -> float:
    """Return whether at least one of the first ``k`` results is relevant.

    For example, if the first three retrieved files are ``["x.py", "y.py",
    "answer.py"]``, ``hit_rate(..., 2)`` is ``0.0`` when only ``answer.py`` is
    gold, while ``hit_rate(..., 3)`` is ``1.0``.  Chunk-level hits use the same
    inclusive range-overlap rule as :func:`recall_at_k`.
    """
    _validate_k(k)
    _validate_level(level)
    return float(any(_is_relevant(item, gold, level) for item in retrieved[:k]))


def _is_relevant(item: Any, gold: Sequence[Any], level: EvaluationLevel) -> bool:
    if not gold:
        return False
    if level == "file":
        return _file_path(item) in {_file_path(gold_item) for gold_item in gold}
    item_range = _line_range(item)
    return any(_ranges_overlap(item_range, _line_range(gold_item)) for gold_item in gold)


def _file_path(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("file_path", "file"):
            if key in item:
                return _string_value(item[key], key)
    for attribute in ("file_path", "file"):
        if hasattr(item, attribute):
            return _string_value(getattr(item, attribute), attribute)
    raise TypeError("file-level items must be paths or expose file_path/file")


def _line_range(item: Any) -> tuple[str, int, int]:
    path: Any
    start: Any
    end: Any
    if isinstance(item, Mapping):
        path = item.get("file_path", item.get("file"))
        start = item.get("start_line", item.get("start"))
        end = item.get("end_line", item.get("end"))
    else:
        path = getattr(item, "file_path", getattr(item, "file", None))
        start = getattr(item, "start_line", getattr(item, "start", None))
        end = getattr(item, "end_line", getattr(item, "end", None))

    path = _string_value(path, "file_path/file")
    if not isinstance(start, int) or isinstance(start, bool):
        raise TypeError("chunk ranges must have an integer start line")
    if not isinstance(end, int) or isinstance(end, bool):
        raise TypeError("chunk ranges must have an integer end line")
    if start < 1 or end < start:
        raise ValueError("chunk ranges must be one-based and end at or after start")
    return path, start, end


def _ranges_overlap(
    first: tuple[str, int, int], second: tuple[str, int, int]
) -> bool:
    return first[0] == second[0] and first[1] <= second[2] and second[1] <= first[2]


def _string_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validate_level(level: str) -> None:
    if level not in ("file", "chunk"):
        raise ValueError("level must be 'file' or 'chunk'")
