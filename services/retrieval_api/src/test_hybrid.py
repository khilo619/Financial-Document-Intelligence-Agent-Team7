from src.bm25_engine import BM25Engine
from src.qdrant_store import QdrantStore
from src.rrf_fusion import RRFFusion


def test_hybrid_retrieval():
    query = "Finished Goods"

    # ---------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------

    qdrant = QdrantStore()

    dense_results = qdrant.search(
        query=query,
        top_k=3,
    )

    assert isinstance(dense_results, list)

    # ---------------------------------------------------------
    # Sparse retrieval
    # ---------------------------------------------------------

    documents = [
        {
            "chunk_id": "test-cts-2019-001",
            "document_id": "cts-corporation_2019.pdf",
            "content": (
                "| Category | 2019 | 2018 |\n"
                "| Finished Goods | 9,447 | 8,912 |"
            ),
        },
        {
            "chunk_id": "test-jabil-2019-001",
            "document_id": "jabil-circuit-inc_2019.pdf",
            "content": (
                "| Category | 2019 | 2018 |\n"
                "| Finished Goods | 314,258 | 289,114 |"
            ),
        },
        {
            "chunk_id": "test-cts-revenue-2019",
            "document_id": "cts-corporation_2019.pdf",
            "content": (
                "| Revenue | 2019 | 2018 |\n"
                "| Net Sales | 1,000 | 950 |"
            ),
        },
    ]

    bm25 = BM25Engine()

    bm25.index_documents(documents)

    sparse_results = bm25.search(
        query=query,
        top_k=3,
    )

    assert isinstance(sparse_results, list)
    assert len(sparse_results) > 0

    # ---------------------------------------------------------
    # RRF fusion
    # ---------------------------------------------------------

    rrf = RRFFusion(k=60)

    fused_results = rrf.fuse(
        [
            dense_results,
            sparse_results,
        ]
    )

    assert isinstance(fused_results, list)
    assert len(fused_results) > 0

    for result in fused_results:
        assert "chunk_id" in result
        assert "rrf_score" in result

    # RRF output should be sorted by score.
    scores = [
        result["rrf_score"]
        for result in fused_results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )