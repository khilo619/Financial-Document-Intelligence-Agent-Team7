"""
doc_processor_api: Layout OCR and Table Parsing microservice.
Owned by Zeina (Member 1).
"""

import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from shared.config import ServiceName
from shared.models import ProcessPdfRequest, ProcessPdfResponse

try:
    from services.doc_processor_api.src.ocr import (
        get_doc_converter,
        process_pdfs_to_custom_schema,
    )
except ImportError:
    from src.ocr import (
        get_doc_converter,
        process_pdfs_to_custom_schema,
    )

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DocProcessorAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Docling model weights ONCE when the API starts up
    try:
        logger.info("Initializing Docling DocumentConverter and loading weights...")
        app.state.doc_converter = get_doc_converter()
        logger.info("Docling model weights successfully loaded into memory!")
    except (RuntimeError, ImportError, ModuleNotFoundError, OSError, ValueError) as e:
        logger.warning(
            "Docling DocumentConverter deferred (offline/CI environment): %s", e
        )
        app.state.doc_converter = None

    yield

    # Clean up when server shuts down
    if getattr(app.state, "doc_converter", None) is not None:
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

    # Retrieve the pre-loaded converter instance from app.state
    doc_converter = getattr(raw_request.app.state, "doc_converter", None)
    if doc_converter is None:
        doc_converter = get_doc_converter()

    try:
        blocks = process_pdfs_to_custom_schema(
            pdf_path=request.pdf_path,
            doc_id=doc_id,
            doc_converter=doc_converter,
        )
    except Exception as e:
        logger.exception("Failed to process PDF %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"PDF processing failed: {e}"
        ) from e

    total_pages = max((b.page for b in blocks), default=1) if blocks else 1
    elapsed = round(time.time() - start_time, 3)

    return ProcessPdfResponse(
        document_id=doc_id,
        total_pages=total_pages,
        total_blocks=len(blocks),
        blocks=blocks,
        processing_time_s=elapsed,
    )


@app.post("/upload_pdf", response_model=ProcessPdfResponse)
async def upload_pdf(
    raw_request: Request,
    file: Annotated[UploadFile, File(...)],
    document_id: Annotated[str | None, Form()] = None,
):
    """
    Accepts a multipart file upload of a PDF and extracts layout blocks.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be a valid .pdf document"
        )

    clean_doc_id = document_id or Path(file.filename).stem
    start_time = time.time()
    logger.info(
        "Processing uploaded PDF: %s (filename: %s)", clean_doc_id, file.filename
    )

    doc_converter = getattr(raw_request.app.state, "doc_converter", None)
    if doc_converter is None:
        doc_converter = get_doc_converter()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        blocks = process_pdfs_to_custom_schema(
            pdf_path=tmp_path,
            doc_id=clean_doc_id,
            doc_converter=doc_converter,
        )
    except Exception as e:
        logger.exception("Failed to process uploaded PDF %s", clean_doc_id)
        raise HTTPException(
            status_code=500, detail=f"PDF processing failed: {e}"
        ) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    total_pages = max((b.page for b in blocks), default=1) if blocks else 1
    elapsed = round(time.time() - start_time, 3)

    return ProcessPdfResponse(
        document_id=clean_doc_id,
        total_pages=total_pages,
        total_blocks=len(blocks),
        blocks=blocks,
        processing_time_s=elapsed,
    )
