"""
doc_processor_api: Layout OCR and Table Parsing microservice.
Owned by Zeina (Member 1) - Initial scaffold by Khaled (Repo Lead).
"""

import time
import logging
from fastapi import FastAPI, HTTPException
from services.doc_processor_api.src.ocr import process_pdfs_to_custom_schema
from models import ProcessPdfRequest, ProcessPdfResponse, ServiceName

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
    
    # Standardize document_id fallback
    doc_id = request.document_id or request.pdf_path.split("/")[-1]
    logger.info("Processing PDF document: %s", doc_id)

    try:
        blocks = process_pdfs_to_custom_schema(request.pdf_path, doc_id)
    except Exception as e:
        logger.error("Failed to process PDF %s: %s", doc_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


    total_pages = max((b.page for b in blocks), default=0) + 1 if blocks else 1
    elapsed = round(time.time() - start_time, 3)

    return ProcessPdfResponse(
        document_id=doc_id,
        total_pages=total_pages,
        total_blocks=len(blocks),
        blocks=blocks,
        processing_time_s=elapsed,
    )
