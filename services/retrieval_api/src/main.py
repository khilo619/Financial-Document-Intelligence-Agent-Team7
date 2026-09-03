"""
retrieval_api: Hybrid Dense + Sparse Search and Cross-Encoder Reranking microservice.
Owned by Salma (Member 2) - Initial scaffold by Khaled (Repo Lead).
"""

import logging
import time

from fastapi import FastAPI

from shared.config import ServiceName
from shared.models import RetrievedChunk, SearchQueryRequest, SearchQueryResponse

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("RetrievalAPI")

app = FastAPI(
    title="Project LEDGER - Hybrid Retrieval API",
    description="Dense (Qdrant) + Sparse (BM25) search fused via RRF and reranked via Cross-Encoder.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": ServiceName.RETRIEVAL.value,
        "port": 8003,
    }


@app.post("/search", response_model=SearchQueryResponse)
def search_documents(request: SearchQueryRequest):
    """
    Executes hybrid retrieval followed by cross-encoder reranking.
    """
    start_time = time.time()
    logger.info(
        "Executing search query: '%s' (top_k=%d, top_n=%d)",
        request.query,
        request.top_k,
        request.top_n,
    )

    # Day 1 Scaffold: Return dummy reranked candidates matching RetrievedChunk schema
    mock_chunks = [
        RetrievedChunk(
            chunk_id="chk-cts-2019-01",
            document_id="cts-corporation_2019.pdf",
            page=1,
            section="Note 4: Inventories",
            content_type="table",
            content="| Finished Goods | 2019: $9,447 | 2018: $8,912 | (In thousands) |",
            score=0.942,
            dense_score=0.881,
            sparse_score=15.34,
            rerank_score=0.985,
            bbox=[50.0, 90.0, 500.0, 200.0],
            metadata={"scale": "thousand", "company": "CTS Corporation", "year": 2019},
        ),
        RetrievedChunk(
            chunk_id="chk-jabil-2019-01",
            document_id="jabil-circuit-inc_2019.pdf",
            page=1,
            section="Inventories",
            content_type="table",
            content="| Finished Goods | 2019: $314,258 | 2018: $289,114 | (In thousands) |",
            score=0.915,
            dense_score=0.842,
            sparse_score=14.12,
            rerank_score=0.962,
            bbox=[45.0, 80.0, 510.0, 190.0],
            metadata={
                "scale": "thousand",
                "company": "Jabil Circuit Inc",
                "year": 2019,
            },
        ),
    ]

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return SearchQueryResponse(
        query=request.query,
        results=mock_chunks[: request.top_n],
        total_found=len(mock_chunks),
        execution_time_ms=elapsed_ms,
    )
