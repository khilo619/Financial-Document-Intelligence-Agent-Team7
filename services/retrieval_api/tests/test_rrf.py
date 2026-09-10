from src.rrf_fusion import RRFFusion


def test_rrf_fusion():
    dense_results = [
        {
            "chunk_id": "A",
            "content": "CTS Finished Goods",
            "score": 0.90,
        },
        {
            "chunk_id": "B",
            "content": "CTS Revenue",
            "score": 0.85,
        },
        {
            "chunk_id": "C",
            "content": "Jabil Finished Goods",
            "score": 0.80,
        },
    ]

    bm25_results = [
        {
            "chunk_id": "C",
            "content": "Jabil Finished Goods",
            "score": 12.5,
        },
        {
            "chunk_id": "A",
            "content": "CTS Finished Goods",
            "score": 10.2,
        },
        {
            "chunk_id": "D",
            "content": "CTS Revenue",
            "score": 5.1,
        },
    ]

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [
            dense_results,
            bm25_results,
        ]
    )

    assert len(results) == 4

    chunk_ids = [result["chunk_id"] for result in results]

    assert set(chunk_ids) == {"A", "B", "C", "D"}

    # A and C appear in both retrieval systems,
    # so they should receive contributions from both lists.
    result_a = next(result for result in results if result["chunk_id"] == "A")

    result_c = next(result for result in results if result["chunk_id"] == "C")

    assert result_a["dense_score"] == 0.90
    assert result_a["sparse_score"] == 10.2

    assert result_c["dense_score"] == 0.80
    assert result_c["sparse_score"] == 12.5

    assert result_a["rrf_score"] > 0
    assert result_c["rrf_score"] > 0


def test_rrf_results_are_sorted():
    dense_results = [
        {"chunk_id": "A", "content": "A"},
        {"chunk_id": "B", "content": "B"},
    ]

    sparse_results = [
        {"chunk_id": "B", "content": "B"},
        {"chunk_id": "A", "content": "A"},
    ]

    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        [
            dense_results,
            sparse_results,
        ]
    )

    scores = [result["rrf_score"] for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )
