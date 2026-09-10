from src.reranker import Reranker


def test_reranker():
    results = [
        {
            "chunk_id": "test-cts-2019-001",
            "document_id": "cts-corporation_2019.pdf",
            "content": ("| Category | 2019 | 2018 |\n| Finished Goods | 9,447 | 8,912 |"),
            "rrf_score": 0.032522,
        },
        {
            "chunk_id": "test-cts-revenue-2019",
            "document_id": "cts-corporation_2019.pdf",
            "content": ("| Revenue | 2019 | 2018 |\n| Net Sales | 1,000 | 950 |"),
            "rrf_score": 0.016393,
        },
        {
            "chunk_id": "test-jabil-2019-001",
            "document_id": "jabil-circuit-inc_2019.pdf",
            "content": ("| Category | 2019 | 2018 |\n| Finished Goods | 314,258 | 289,114 |"),
            "rrf_score": 0.015873,
        },
    ]

    query = "Finished goods balance for CTS in 2019"

    reranker = Reranker()

    reranked_results = reranker.rerank(
        query=query,
        results=results,
        top_n=3,
    )

    assert len(reranked_results) == 3

    for result in reranked_results:
        assert "rerank_score" in result
        assert isinstance(
            result["rerank_score"],
            float,
        )

    scores = [result["rerank_score"] for result in reranked_results]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    assert all("chunk_id" in result for result in reranked_results)

    assert all("content" in result for result in reranked_results)

    assert len({result["chunk_id"] for result in reranked_results}) == 3
