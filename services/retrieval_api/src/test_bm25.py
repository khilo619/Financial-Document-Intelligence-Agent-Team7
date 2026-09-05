from src.bm25_engine import BM25Engine


def test_bm25_search():
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

    engine = BM25Engine()
    engine.index_documents(documents)

    results = engine.search(
        query="Finished Goods",
        top_k=3,
    )

    assert len(results) > 0

    result_ids = {
    result["chunk_id"]
    for result in results
    }

    assert {
      "test-cts-2019-001",
      "test-jabil-2019-001",
    }.issubset(result_ids)

def test_bm25_filter():
    documents = [
        {
            "chunk_id": "table-001",
            "document_id": "cts-2019.pdf",
            "content": "Finished Goods 9,447",
            "content_type": "table",
            "metadata": {
                "company": "CTS Corporation",
                "year": 2019,
            },
        },
        {
            "chunk_id": "text-001",
            "document_id": "cts-2019.pdf",
            "content": "Revenue increased strongly.",
            "content_type": "text",
            "metadata": {
                "company": "CTS Corporation",
                "year": 2019,
            },
        },
    ]

    engine = BM25Engine()
    engine.index_documents(documents)

    results = engine.search(
        query="Finished Goods",
        top_k=10,
        filters={
            "company": "CTS Corporation",
            "year": 2019,
            "content_type": "table",
        },
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "table-001"
    assert results[0]["content_type"] == "table"