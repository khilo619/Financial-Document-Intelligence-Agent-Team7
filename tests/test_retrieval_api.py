"""
Tests for retrieval_api components: BM25Engine and RRFFusion edge cases.
"""

import pytest

pytest.importorskip("rank_bm25")

from services.retrieval_api.src.bm25_engine import BM25Engine
from services.retrieval_api.src.rrf_fusion import RRFFusion


def test_bm25_empty_documents_does_not_crash():
    """Verify BM25Engine handles an empty document collection gracefully (fresh Qdrant boot)."""
    engine = BM25Engine()
    engine.index_documents([])
    assert engine.bm25 is None
    results = engine.search("revenue", top_k=5)
    assert results == []


def test_bm25_indexing_and_search():
    """Verify BM25Engine indexes documents and returns ranked results with metadata filtering."""
    engine = BM25Engine()
    docs = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_apple",
            "content": "Apple recorded total revenue of 100 billion dollars in fiscal year 2023.",
            "content_type": "text",
            "page": 1,
            "metadata": {"company": "Apple", "year": "2023", "section": "Financials"},
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "doc_microsoft",
            "content": "Microsoft reported strong cloud computing growth with Azure.",
            "content_type": "text",
            "page": 2,
            "metadata": {"company": "Microsoft", "year": "2023", "section": "Cloud"},
        },
    ]
    engine.index_documents(docs)
    assert engine.bm25 is not None

    # Search for Apple revenue
    results = engine.search("revenue", top_k=5)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "chunk_1"

    # Search with filter for Microsoft returns chunk_2 with score 0.0
    filtered = engine.search("revenue", top_k=5, filters={"company": "Microsoft"})
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "chunk_2"
    assert filtered[0]["score"] == 0.0


def test_rrf_fusion_empty_results():
    """Verify RRFFusion handles empty result lists without errors."""
    fusion = RRFFusion()
    fused = fusion.fuse([[], []])
    assert fused == []


def test_rrf_fusion_combines_ranks():
    """Verify RRFFusion properly calculates reciprocal rank scores."""
    fusion = RRFFusion(k=60)
    dense = [{"chunk_id": "c1", "score": 0.9, "content": "test 1"}]
    sparse = [{"chunk_id": "c1", "score": 12.5, "content": "test 1"}]
    fused = fusion.fuse([dense, sparse])
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "c1"
    assert fused[0]["dense_score"] == 0.9
    assert fused[0]["sparse_score"] == 12.5
    # score = 1/(60+1) + 1/(60+1) = 2/61
    assert pytest.approx(fused[0]["rrf_score"], 0.0001) == (2.0 / 61.0)
