"""
doc_processor_api: Layout OCR and Table Parsing microservice.
Owned by Zeina (Member 1) - Initial scaffold by Khaled (Repo Lead).
"""

import logging
import time

from fastapi import FastAPI

from shared.config import ServiceName
from shared.models import DocumentBlock, ProcessPdfRequest, ProcessPdfResponse

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DocProcessorAPI")

app = FastAPI(
    title="Project LEDGER - Document Processor API",
    description="Extracts Markdown, tables, and bounding boxes from raw financial PDFs.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": ServiceName.DOC_PROCESSOR.value,
        "port": 8002,
    }


@app.post("/process_pdf", response_model=ProcessPdfResponse)
def process_pdf(request: ProcessPdfRequest):
    """
    Parses a PDF into layout-aware blocks (text, markdown tables, bounding boxes).
    """
    start_time = time.time()
    doc_id = request.document_id or request.pdf_path.split("/")[-1]
    logger.info("Processing PDF document: %s", doc_id)

    # Day 1 Scaffold: Return dummy layout-aware blocks conforming to DocumentBlock schema
    sample_blocks = [
        DocumentBlock(
            block_id=f"{doc_id}-blk-001",
            document_id=doc_id,
            page=1,
            content_type="header",
            markdown_content="# CTS CORPORATION - NOTE 4: INVENTORIES",
            bbox=[50.0, 50.0, 500.0, 80.0],
            metadata={"period": "2019"},
        ),
        DocumentBlock(
            block_id=f"{doc_id}-blk-002",
            document_id=doc_id,
            page=1,
            content_type="table",
            markdown_content="| Finished Goods | 2019: $9,447 | 2018: $8,912 | (In thousands) |",
            table_rows=[
                ["Category", "2019", "2018"],
                ["Finished Goods", "9,447", "8,912"],
            ],
            bbox=[50.0, 90.0, 500.0, 200.0],
            metadata={"scale": "thousand", "period": "2019"},
        ),
    ]

    elapsed = round(time.time() - start_time, 3)
    return ProcessPdfResponse(
        document_id=doc_id,
        total_pages=1,
        total_blocks=len(sample_blocks),
        blocks=sample_blocks,
        processing_time_s=elapsed,
    )
