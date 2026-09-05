from src.reranker import Reranker


def main():
    # Example candidates produced by the RRF stage.
    results = [
        {
            "chunk_id": "test-cts-2019-001",
            "document_id": "cts-corporation_2019.pdf",
            "content": (
                "| Category | 2019 | 2018 |\n"
                "| Finished Goods | 9,447 | 8,912 |"
            ),
            "rrf_score": 0.032522,
        },
        {
            "chunk_id": "test-cts-revenue-2019",
            "document_id": "cts-corporation_2019.pdf",
            "content": (
                "| Revenue | 2019 | 2018 |\n"
                "| Net Sales | 1,000 | 950 |"
            ),
            "rrf_score": 0.016393,
        },
        {
            "chunk_id": "test-jabil-2019-001",
            "document_id": "jabil-circuit-inc_2019.pdf",
            "content": (
                "| Category | 2019 | 2018 |\n"
                "| Finished Goods | 314,258 | 289,114 |"
            ),
            "rrf_score": 0.015873,
        },
    ]

    query = "Finished goods balance for CTS in 2019"

    # Initialize the Cross-Encoder reranker.
    reranker = Reranker()

    # Rerank the candidate documents.
    reranked_results = reranker.rerank(
        query=query,
        results=results,
        top_n=3,
    )

    print("\n===== CROSS-ENCODER RERANKING =====")

    for rank, result in enumerate(reranked_results, start=1):
        print(
            f"Rank {rank} | "
            f"Chunk: {result['chunk_id']} | "
            f"RRF Score: {result['rrf_score']:.6f} | "
            f"Rerank Score: {result['rerank_score']:.6f}"
        )


if __name__ == "__main__":
    main()