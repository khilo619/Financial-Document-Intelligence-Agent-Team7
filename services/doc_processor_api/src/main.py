"""
doc_processor_api: Layout OCR and Table Parsing microservice.
Owned by Zeina (Member 1) - Initial scaffold by Khaled (Repo Lead).
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request

from services.doc_processor_api.src.ocr import (
    process_pdfs_to_custom_schema,
    get_doc_converter
)
from models import ProcessPdfRequest, ProcessPdfResponse, ServiceName

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DocProcessorAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Docling model weights ONCE when the API starts up
    logger.info("Initializing Docling DocumentConverter and loading weights...")
    app.state.doc_converter = get_doc_converter()
    logger.info("Docling model weights successfully loaded into memory!")
    
    yield
    
    # Clean up when server shuts down
    del app.state.doc_converter


app = FastAPI(
    title="Project LEDGER - Document Processor API",
    description="Extracts Markdown, tables, and bounding boxes from raw financial PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": ServiceName.DOC_PROCESSOR.value,
        "port": 8002,
    }


@app.post("/process_pdf", response_model=ProcessPdfResponse)
def process_pdf(request: ProcessPdfRequest, raw_request: Request):
    """
    Parses a PDF into layout-aware blocks (text, markdown tables, bounding boxes).
    """
    start_time = time.time()
    doc_id = request.document_id or request.pdf_path.split("/")[-1]
    logger.info("Processing PDF document: %s", doc_id)

    # 2. Retrieve the pre-loaded converter instance from app.state
    doc_converter = raw_request.app.state.doc_converter
    

    try:
        # Pass converter into processing function
        blocks = process_pdfs_to_custom_schema(
    pdf_path=request.pdf_path, 
    doc_id=doc_id, 
    doc_converter=doc_converter
)
    except Exception as e:
        logger.error("Failed to process PDF %s: %s", doc_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

    total_pages = max((b.page for b in blocks), default=1) if blocks else 1
    elapsed = round(time.time() - start_time, 3)

    return ProcessPdfResponse(
        document_id=doc_id,
        total_pages=total_pages,
        total_blocks=len(blocks),
        blocks=blocks,
        processing_time_s=elapsed,
    )
