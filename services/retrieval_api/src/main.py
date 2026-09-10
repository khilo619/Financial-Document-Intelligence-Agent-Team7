"""
retrieval_api: Hybrid Dense + Sparse Search and Cross-Encoder Reranking microservice.
Owned by Salma (Member 2) - Initial scaffold by Khaled (Repo Lead).
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from shared.config import ServiceName
from shared.models import (
    IndexBlocksRequest,
    IndexBlocksResponse,
    RetrievedChunk,
    SearchQueryRequest,
    SearchQueryResponse,
)

from .bm25_engine import BM25Engine
from .qdrant_store import QdrantStore
from .reranker import Reranker
from .rrf_fusion import RRFFusion

# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)

logger = logging.getLogger("RetrievalAPI")


# ---------------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Startup:
        - Initialize Qdrant store and embedding model
        - Initialize BM25 engine
        - Initialize RRF fusion
        - Initialize Cross-Encoder reranker
        - Build BM25 index

    Shutdown:
        - Perform cleanup if required
    """

    logger.info("Starting Retrieval API...")

    # -----------------------------------------------------
    # Initialize retrieval components
    # -----------------------------------------------------

    app.state.qdrant_store = QdrantStore()

    app.state.bm25_engine = BM25Engine()

    app.state.rrf_fusion = RRFFusion()

    app.state.reranker = Reranker()

    # -----------------------------------------------------
    # Initialize BM25 index
    # -----------------------------------------------------

    logger.info("Loading documents from Qdrant...")

    documents = app.state.qdrant_store.get_all_documents()

    app.state.bm25_engine.index_documents(documents)

    logger.info(
        "BM25 index initialized with %d documents",
        len(documents),
    )

    logger.info("Retrieval API is ready.")

    # -----------------------------------------------------
    # Application runs here
    # -----------------------------------------------------

    yield

    # -----------------------------------------------------
    # Shutdown
    # -----------------------------------------------------

    logger.info("Shutting down Retrieval API...")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title=("Project LEDGER - Hybrid Retrieval API"),
    description=("Dense (Qdrant) + Sparse (BM25) search fused via RRF and reranked via Cross-Encoder."),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------


@app.get("/health")
def health_check():
    """
    Health check endpoint used by Docker
    and other services.
    """

    return {
        "status": "healthy",
        "service": ServiceName.RETRIEVAL.value,
        "port": 8003,
    }


@app.get("/stats")
def get_retrieval_stats(request: Request):
    """
    Returns live indexing statistics for Qdrant vector database and BM25 engine.
    """
    qdrant_store = getattr(request.app.state, "qdrant_store", None)
    bm25_engine = getattr(request.app.state, "bm25_engine", None)

    qdrant_stats = qdrant_store.get_collection_stats() if qdrant_store else {"points_count": 0, "status": "unavailable"}
    bm25_count = len(bm25_engine.documents) if bm25_engine and hasattr(bm25_engine, "documents") else 0

    return {
        "status": "healthy",
        "bm25_documents_count": bm25_count,
        "qdrant": qdrant_stats,
    }


# ---------------------------------------------------------
# Index blocks endpoint
# ---------------------------------------------------------


@app.post(
    "/index_blocks",
    response_model=IndexBlocksResponse,
)
def index_document_blocks(
    index_request: IndexBlocksRequest,
    request: Request,
):
    """
    Index DocumentBlocks into Qdrant vector store and refresh in-memory BM25 index.
    """
    logger.info(
        "Received indexing request for document '%s' with %d blocks",
        index_request.document_id,
        len(index_request.blocks),
    )

    qdrant_store = request.app.state.qdrant_store
    bm25_engine = request.app.state.bm25_engine

    indexed_count = qdrant_store.index_blocks(index_request.blocks)

    # Refresh BM25 index with updated Qdrant corpus
    updated_documents = qdrant_store.get_all_documents()
    bm25_engine.index_documents(updated_documents)

    logger.info(
        "Successfully indexed %d blocks for '%s'. Total BM25 index size: %d documents.",
        indexed_count,
        index_request.document_id,
        len(updated_documents),
    )

    return IndexBlocksResponse(
        status="indexed",
        document_id=index_request.document_id,
        indexed_chunks=indexed_count,
    )


# ---------------------------------------------------------
# Hybrid search endpoint
# ---------------------------------------------------------


@app.post(
    "/search",
    response_model=SearchQueryResponse,
)
def search_documents(
    search_request: SearchQueryRequest,
    request: Request,
):
    """
    Execute hybrid retrieval:

        Query
          ↓
        Dense Search (Qdrant)
          +
        Sparse Search (BM25)
          ↓
        RRF Fusion
          ↓
        Top-K Candidates
          ↓
        Cross-Encoder Reranking
          ↓
        Top-N Results
    """

    # -----------------------------------------------------
    # Start total latency measurement
    # -----------------------------------------------------

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # Get initialized components
    # -----------------------------------------------------

    qdrant_store = request.app.state.qdrant_store

    bm25_engine = request.app.state.bm25_engine

    rrf_fusion = request.app.state.rrf_fusion

    reranker = request.app.state.reranker

    logger.info(
        "Executing search query: '%s' (top_k=%d, top_n=%d, filters=%s, reranking=%s)",
        search_request.query,
        search_request.top_k,
        search_request.top_n,
        search_request.filters,
        search_request.use_reranking,
    )

    # =====================================================
    # 1. Dense retrieval
    # =====================================================

    stage_start = time.perf_counter()

    qdrant_results = qdrant_store.search(
        query=search_request.query,
        top_k=search_request.top_k,
        filters=search_request.filters,
    )

    dense_latency_ms = (time.perf_counter() - stage_start) * 1000

    logger.info(
        "Dense search returned %d results in %.2f ms",
        len(qdrant_results),
        dense_latency_ms,
    )

    # =====================================================
    # 2. Sparse retrieval
    # =====================================================

    stage_start = time.perf_counter()

    bm25_results = bm25_engine.search(
        query=search_request.query,
        top_k=search_request.top_k,
        filters=search_request.filters,
    )

    sparse_latency_ms = (time.perf_counter() - stage_start) * 1000

    logger.info(
        "Sparse search returned %d results in %.2f ms",
        len(bm25_results),
        sparse_latency_ms,
    )

    # =====================================================
    # 3. RRF Fusion
    # =====================================================

    stage_start = time.perf_counter()

    fused_results = rrf_fusion.fuse(
        [
            qdrant_results,
            bm25_results,
        ]
    )

    rrf_latency_ms = (time.perf_counter() - stage_start) * 1000

    logger.info(
        "Dense results: %d | Sparse results: %d | Fused results: %d | RRF latency: %.2f ms",
        len(qdrant_results),
        len(bm25_results),
        len(fused_results),
        rrf_latency_ms,
    )

    # =====================================================
    # 4. Select Top-K candidates
    # =====================================================

    candidates = fused_results[: search_request.top_k]

    logger.info(
        "Selected %d candidates for reranking",
        len(candidates),
    )

    # =====================================================
    # 5. Cross-Encoder reranking
    # =====================================================

    rerank_latency_ms = 0.0

    if search_request.use_reranking and candidates:
        logger.info(
            "Running Cross-Encoder reranking on %d candidates",
            len(candidates),
        )

        stage_start = time.perf_counter()

        final_results = reranker.rerank(
            query=search_request.query,
            results=candidates,
            top_n=search_request.top_n,
        )

        rerank_latency_ms = (time.perf_counter() - stage_start) * 1000

        logger.info(
            "Reranking completed in %.2f ms",
            rerank_latency_ms,
        )

    else:
        logger.info("Cross-Encoder reranking disabled")

        final_results = candidates[: search_request.top_n]

    # =====================================================
    # 6. Convert results to RetrievedChunk
    # =====================================================

    results = []

    for result in final_results:
        metadata = result.get(
            "metadata",
            {},
        )

        # -------------------------------------------------
        # Determine final score
        # -------------------------------------------------

        if search_request.use_reranking and result.get("rerank_score") is not None:
            final_score = result["rerank_score"]

        else:
            final_score = result["rrf_score"]

        # -------------------------------------------------
        # Build response object
        # -------------------------------------------------

        results.append(
            RetrievedChunk(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                page=result["page"],
                section=metadata.get(
                    "section",
                    "",
                ),
                content_type=result["content_type"],
                content=result["content"],
                score=final_score,
                dense_score=result.get("dense_score"),
                sparse_score=result.get("sparse_score"),
                rerank_score=result.get("rerank_score"),
                bbox=result.get("bbox"),
                metadata=metadata,
            )
        )

    # =====================================================
    # 7. Total execution time
    # =====================================================

    elapsed_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    # =====================================================
    # 8. Latency logging
    # =====================================================

    logger.info(
        "Latency breakdown | Dense: %.2f ms | Sparse: %.2f ms | RRF: %.2f ms | Rerank: %.2f ms | Total: %.2f ms",
        dense_latency_ms,
        sparse_latency_ms,
        rrf_latency_ms,
        rerank_latency_ms,
        elapsed_ms,
    )

    logger.info(
        "Search completed: %d results in %.2f ms",
        len(results),
        elapsed_ms,
    )

    # =====================================================
    # 9. API response
    # =====================================================

    return SearchQueryResponse(
        query=search_request.query,
        results=results,
        total_found=len(fused_results),
        execution_time_ms=elapsed_ms,
    )
