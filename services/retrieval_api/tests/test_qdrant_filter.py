from shared.models import DocumentBlock
from src.qdrant_store import QdrantStore


def test_qdrant_content_type_filter():
    store = QdrantStore()

    table_block = DocumentBlock(
        block_id="pytest-filter-table-001",
        document_id="pytest-filter.pdf",
        page=1,
        content_type="table",
        markdown_content=("| Category | 2019 | 2018 |\n| Finished Goods | 9,447 | 8,912 |"),
        metadata={
            "company": "Pytest Company",
            "year": 2019,
            "section": "Inventories",
        },
    )

    text_block = DocumentBlock(
        block_id="pytest-filter-text-001",
        document_id="pytest-filter.pdf",
        page=2,
        content_type="text",
        markdown_content=("The company reported strong revenue growth in 2019."),
        metadata={
            "company": "Pytest Company",
            "year": 2019,
            "section": "Management Discussion",
        },
    )

    store.index_block(table_block)
    store.index_block(text_block)

    table_results = store.search(
        query="Finished Goods",
        top_k=10,
        filters={
            "content_type": "table",
        },
    )

    assert len(table_results) > 0

    assert all(result["content_type"] == "table" for result in table_results)

    assert any(result["chunk_id"] == "pytest-filter-table-001" for result in table_results)

    text_results = store.search(
        query="revenue growth",
        top_k=10,
        filters={
            "content_type": "text",
        },
    )

    assert len(text_results) > 0

    assert all(result["content_type"] == "text" for result in text_results)

    assert any(result["chunk_id"] == "pytest-filter-text-001" for result in text_results)
