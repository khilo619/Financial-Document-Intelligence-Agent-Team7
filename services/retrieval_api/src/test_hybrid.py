from src.bm25_engine import BM25Engine
from src.qdrant_store import QdrantStore
from src.rrf_fusion import RRFFusion


def main():
    # The query used by both retrieval systems.
    query = "Finished Goods"

    # ---------------------------------------------------------
    # 1. Dense retrieval using Qdrant
    # ---------------------------------------------------------

    qdrant = QdrantStore()

    dense_results = qdrant.search(
        query=query,
        top_k=3,
    )

    print("\n===== DENSE SEARCH (Qdrant) =====")

    for rank, result in enumerate(dense_results, start=1):
        print(
            f"Rank {rank} | "
            f"Chunk: {result['chunk_id']} | "
            f"Score: {result['score']:.6f}"
        )

    # ---------------------------------------------------------
    # 2. Sparse retrieval using BM25
    # ---------------------------------------------------------

    documents = [
        {
            "chunk_id": "test-cts-2019-001",
            "document_id": "cts-corporation_2019.pdf",
            "content": "| Category | 2019 | 2018 |\n| Finished Goods | 9,447 | 8,912 |",
        },
        {
            "chunk_id": "test-jabil-2019-001",
            "document_id": "jabil-circuit-inc_2019.pdf",
            "content": "| Category | 2019 | 2018 |\n| Finished Goods | 314,258 | 289,114 |",
        },
        {
            "chunk_id": "test-cts-revenue-2019",
            "document_id": "cts-corporation_2019.pdf",
            "content": "| Revenue | 2019 | 2018 |\n| Net Sales | 1,000 | 950 |",
        },
    ]

    bm25 = BM25Engine()

    bm25.index_documents(documents)

    sparse_results = bm25.search(
        query=query,
        top_k=3,
    )

    print("\n===== SPARSE SEARCH (BM25) =====")

    for rank, result in enumerate(sparse_results, start=1):
        print(
            f"Rank {rank} | "
            f"Chunk: {result['chunk_id']} | "
            f"Score: {result['score']:.6f}"
        )

    # ---------------------------------------------------------
    # 3. Reciprocal Rank Fusion
    # ---------------------------------------------------------

    rrf = RRFFusion(k=60)

    fused_results = rrf.fuse(
        [
            dense_results,
            sparse_results,
        ]
    )

    print("\n===== RRF FUSION =====")

    for rank, result in enumerate(fused_results, start=1):
        print(
            f"Rank {rank} | "
            f"Chunk: {result['chunk_id']} | "
            f"RRF Score: {result['rrf_score']:.6f}"
        )


if __name__ == "__main__":
    main()