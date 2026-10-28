from retrieval.hybrid import HybridRetriever, rrf


def test_rrf_matches_hand_computed_scores() -> None:
    # beta: 1/62 + 1/61, gamma: 1/63 + 1/62, alpha: 1/61
    assert rrf([["alpha", "beta", "gamma"], ["beta", "gamma"]]) == [
        "beta",
        "gamma",
        "alpha",
    ]


def test_rrf_ignores_duplicates_and_missing_candidates() -> None:
    assert rrf([["a", "a", "b"], ["c", "b"]]) == ["b", "a", "c"]


def test_rrf_empty_rankings_return_empty() -> None:
    assert rrf([[], []]) == []


def test_hybrid_retriever_searches_thirty_candidates_and_fuses_ids() -> None:
    class Embedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["find it"]
            return [[1.0]]

    class VectorStore:
        def query(self, vector: list[float], k: int) -> list[dict[str, str]]:
            assert vector == [1.0]
            assert k == 30
            return [{"id": "shared"}, {"id": "vector-only"}]

    class BM25:
        def search(self, query: str, k: int) -> list[tuple[str, float]]:
            assert query == "find it"
            assert k == 30
            return [("shared", 1.0), ("bm25-only", 0.5)]

    retriever = HybridRetriever(VectorStore(), BM25(), Embedder())

    assert retriever.search("find it", 3) == [
        "shared",
        "vector-only",
        "bm25-only",
    ]
