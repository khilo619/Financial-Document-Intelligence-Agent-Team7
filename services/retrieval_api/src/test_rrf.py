from src.rrf_fusion import RRFFusion


dense_results = [
    {
        "chunk_id": "A",
        "content": "CTS Finished Goods",
        "dense_score": 0.90,
    },
    {
        "chunk_id": "B",
        "content": "CTS Revenue",
        "dense_score": 0.85,
    },
    {
        "chunk_id": "C",
        "content": "Jabil Finished Goods",
        "dense_score": 0.80,
    },
]


bm25_results = [
    {
        "chunk_id": "C",
        "content": "Jabil Finished Goods",
        "sparse_score": 12.5,
    },
    {
        "chunk_id": "A",
        "content": "CTS Finished Goods",
        "sparse_score": 10.2,
    },
    {
        "chunk_id": "D",
        "content": "CTS Revenue",
        "sparse_score": 5.1,
    },
]


fusion = RRFFusion(k=60)

results = fusion.fuse(
    [
        dense_results,
        bm25_results,
    ]
)

print("RRF Results:")

for rank, result in enumerate(results, start=1):
    print(
        f"Rank {rank} | "
        f"Chunk: {result['chunk_id']} | "
        f"RRF Score: {result['rrf_score']:.6f}"
    )