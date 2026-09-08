"""
tests/test_doc_processor.py: Verification tests for doc_processor_api contracts and health endpoint.
"""

from fastapi.testclient import TestClient

from services.doc_processor_api.src.main import app
from shared.config import ServiceName
from shared.models import DocumentBlock, ProcessPdfRequest, ProcessPdfResponse

client = TestClient(app)


def test_doc_processor_health():
    """Verify doc_processor_api /health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == ServiceName.DOC_PROCESSOR.value
    assert data["port"] == 8002


def test_process_pdf_request_response_models():
    """Verify ProcessPdfRequest and ProcessPdfResponse schemas."""
    req = ProcessPdfRequest(pdf_path="/path/to/test.pdf", document_id="doc_123")
    assert req.document_id == "doc_123"

    block = DocumentBlock(
        block_id="blk_001",
        document_id="doc_123",
        page=1,
        content_type="text",
        markdown_content="Sample financial disclosure paragraph.",
        table_rows=None,
        bbox=[40.0, 100.0, 500.0, 120.0],
        metadata={},
    )
    resp = ProcessPdfResponse(
        document_id="doc_123",
        total_pages=1,
        total_blocks=1,
        blocks=[block],
        processing_time_s=1.23,
    )
    assert resp.total_blocks == 1
    assert resp.blocks[0].content_type == "text"


SAMPLE_GOLD_BLOCKS = [
    {
        "block_id": "blk_001",
        "document_id": "4b587f0c528da24c4a28592df1b81ee6",
        "page": 1,
        "content_type": "header",
        "markdown_content": "Gross Profit",
        "table_rows": None,
        "bbox": [41.0, 719.65, 93.13, 727.87],
        "metadata": {},
    },
    {
        "block_id": "blk_002",
        "document_id": "4b587f0c528da24c4a28592df1b81ee6",
        "page": 1,
        "content_type": "text",
        "markdown_content": "Gross profit in fiscal year 2018 increased to $382.3 million, or 17.7 percent of net sales from $300.8 million, or 16.7 percent of net sales for fiscal year 2017.",
        "table_rows": None,
        "bbox": [41.0, 659.65, 551.2, 703.87],
        "metadata": {},
    },
    {
        "block_id": "blk_004",
        "document_id": "4b587f0c528da24c4a28592df1b81ee6",
        "page": 1,
        "content_type": "table",
        "markdown_content": "|                                          | Fiscal Year   | Fiscal Year   |\n|------------------------------------------|---------------|---------------|\n| ($ in millions)                          | 2018          | 2017          |\n| Net sales                                | $ 2,157.7     | $ 1,797.6     |\n| Gross profit                             | $ 382.3       | $ 300.8       |",
        "table_rows": [
            ["($ in millions)", "Fiscal Year.2018", "Fiscal Year.2017"],
            ["Net sales", "$ 2,157.7", "$ 1,797.6"],
            ["Gross profit", "$ 382.3", "$ 300.8"],
        ],
        "bbox": [40.64, 389.96, 549.45, 543.5],
        "metadata": {},
    },
]


def test_gold_standard_blocks_match_canonical_schema():
    """Verify that gold standard output perfectly conforms to DocumentBlock schema."""
    validated_blocks = [DocumentBlock(**b) for b in SAMPLE_GOLD_BLOCKS]
    assert len(validated_blocks) == 3

    # Verify table block exists with 2D rows
    table_block = next((b for b in validated_blocks if b.content_type == "table"), None)
    assert table_block is not None, "Table block blk_004 must be present"
    assert table_block.table_rows is not None
    assert len(table_block.table_rows) == 3
    assert "Net sales" in table_block.markdown_content


def test_upload_pdf_rejects_non_pdf():
    """Verify that /upload_pdf rejects non-pdf files with 400 Bad Request."""
    response = client.post(
        "/upload_pdf",
        files={"file": ("test.txt", b"Hello text file", "text/plain")},
    )
    assert response.status_code == 400
    assert "Uploaded file must be a valid .pdf document" in response.json()["detail"]
