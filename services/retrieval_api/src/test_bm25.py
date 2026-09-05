from src.bm25_engine import BM25Engine


def main():
    # Create sample documents similar to the chunks
    # produced by the document processing service.
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

    # Initialize the BM25 search engine.
    engine = BM25Engine()

    # Build the BM25 index.
    engine.index_documents(documents)

    for document in engine.documents:
        print("\nSEARCH TEXT:")
        print(engine._build_search_text(document))

    # Define a natural-language search query.
    query = "Finished"

    # Search for the most relevant documents.
    results = engine.search(query, top_k=3)

    print(f"Search query: {query}")
    print(f"Number of results: {len(results)}")

    print("\nBM25 search results:")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score: {result['score']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Document ID: {result['document_id']}")
        print(f"Content: {result['content']}")


if __name__ == "__main__":
    main()